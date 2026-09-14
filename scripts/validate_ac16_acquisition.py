"""Bounded production-transport OTA capture and blind production RX replay."""
import argparse
import hashlib
import json
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aetv.config import AETV_MODES
from aetv.modem import StreamingDemodulator, modulate_continuous_chunks
from aetv.settings import StationSettings
from aetv.sdr import SDRCapture, open_pluto, transmit_pluto
from aetv.station import Station, TxEngine
from aetv.analog_av import AC16CompositeSeparator

ROOT: Path
SOURCE: Path
LATENTS: Path


def save(path, data):
    def finite(value):
        if isinstance(value,dict): return {k:finite(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [finite(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    path.write_text(json.dumps(finite(data), indent=2, allow_nan=False)+'\n')


def capture(args):
    if SOURCE is None or LATENTS is None:
        raise ValueError('Capture requires --source and --latents')
    if not -89.75 <= args.tx <= 0 or not 0 < args.rx <= 49.6:
        raise ValueError('Use hardware-bounded TX gain and explicit nonzero RTL gain')
    out = ROOT/args.name
    out.mkdir(exist_ok=False)
    sent = np.load(LATENTS)
    if sent.ndim != 2 or sent.shape[1] != 19200 or not 1 <= len(sent) <= 60:
        raise ValueError('Expected 1-60 AC16 latent GOPs')
    settings = StationSettings(mode='AC16', waveform_mode=args.mode, pluto_tx_gain=args.tx,
                               sdr_frequency_mhz=439, av_microphone_mix=0)
    chunks = modulate_continuous_chunks(sent, 'AC16', total_gops=len(sent))
    if args.mode == 'analog_av':
        t = np.arange(8000)/8000
        # Distinct program tones identify A/V pairing without speech ambiguity.
        voice = np.concatenate([.2*np.sin(2*np.pi*(500+31*i)*t) for i in range(len(sent))])
        chunks = TxEngine(Station(settings))._composite_chunks(chunks, voice, len(sent), capture_microphone=False)
    chunks = list(chunks)
    np.save(out/'waveform.npy', np.concatenate(chunks))
    duration = sum(map(len, chunks))/48000
    report = dict(mode=args.mode, tx_gain_db=args.tx, rx_gain_db=args.rx, frequency_hz=439000000,
                  source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  sent_latents=str(LATENTS), source_gops=len(sent), duration_s=duration,
                  started_unix=time.time(), receivers=[], tx_off_verified=False,
                  code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  provenance=args.provenance,
                  setup='Existing antennas about 10 ft apart; no shared reference; RTL 1002 excluded.',
                  caveat='Gain is hardware attenuation, not calibrated radiated dBm; RTL has no hardware timestamps.')
    processes=[]
    try:
        radio=open_pluto(settings.pluto_uri)
        if radio._ctrl.find_channel('altvoltage1',True).attrs['powerdown'].value != '1':
            raise RuntimeError('Pluto unexpectedly already transmitting')
        del radio
        for serial in ('1000','1001','1003','1004'):
            path=out/f'rtl-{serial}.cu8'
            log=(out/f'rtl-{serial}.log').open('wb')
            command=['rtl_sdr','-d',serial,'-f','439100000','-s','960000','-g',str(args.rx),'-b','65536',str(path)]
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            processes.append((process,log,path))
            report['receivers'].append(dict(serial=serial,command=command))
        deadline=time.monotonic()+10
        while not all(p.exists() and p.stat().st_size >= 1920000 for _,_,p in processes):
            if time.monotonic()>deadline or any(p.poll() is not None for p,_,_ in processes):
                raise RuntimeError('Receivers did not start')
            time.sleep(.05)
        time.sleep(2)
        for row,(_,_,path) in zip(report['receivers'], processes):
            row['samples_before_tx_call']=path.stat().st_size//2
        report['tx_call_unix']=time.time()
        save(out/'capture.json',report)
        report['complete']=transmit_pluto(chunks,48000,settings,threading.Event(),lambda _:None,max_seconds=duration)
        radio=open_pluto(settings.pluto_uri)
        report['tx_off_verified']=(radio._ctrl.find_channel('altvoltage1',True).attrs['powerdown'].value=='1'
                                   and radio.tx_hardwaregain_chan0==-89.75)
        if not report['tx_off_verified']:
            raise RuntimeError('Pluto shutdown verification failed')
        time.sleep(2)
    finally:
        for row,(process,log,path) in zip(report['receivers'],processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait()
            log.close()
            row['returncode']=process.returncode
            row['bytes']=path.stat().st_size if path.exists() else 0
        report['finished_unix']=time.time()
        save(out/'capture.json',report)
    print(args.name,report['complete'],report['tx_off_verified'],flush=True)


def replay(args):
    path=ROOT/args.name
    metadata=json.loads((path/'capture.json').read_text())
    raw=np.memmap(path/f'rtl-{args.serial}.cu8',dtype=np.uint8,mode='r')
    raw=raw[round(args.join*960000)*2:]
    settings=StationSettings(mode='AC16', waveform_mode=metadata['mode'],rx_source='rtlsdr',
                             sdr_auto_correct=not args.manual, sdr_rx_correction_hz=args.correction)
    sent=np.load(metadata['sent_latents'])
    sent_unit=sent/np.linalg.norm(sent,axis=1,keepdims=True)
    events=[];rows=[];timings=[];pcm=[];input_end=[0.]
    demod=StreamingDemodulator('A',continuous=True,mode_name='AC16',boundary_tracking=True,on_debug=events.append)
    separator=AC16CompositeSeparator() if metadata['mode']=='analog_av' else None
    def write(audio):
        pcm.append(audio.copy())
        if separator is not None: _,audio=separator.process(audio)
        for result in demod.feed(audio):
            latent=result.gops_latents[0]
            cosine=sent_unit@latent/max(np.linalg.norm(latent),1e-12)
            index=int(cosine.argmax())
            rows.append(dict(input_end_s=input_end[0],source_gop=index,cosine=float(cosine[index]),
                             snr_db=float(result.snr_db),stream_start_sample=int(result.stream_start_sample)))
            received.append(latent.copy());weights.append(result.gops_weights[0].copy())
    received=[];weights=[]
    cap=SDRCapture(settings,AETV_MODES['AC16'],SimpleNamespace(write=write),on_error=print,
                   on_status=lambda message:events.append(dict(event='sdr_status',input_end_s=input_end[0],message=message)))
    class Input:
        position=0
        def get(self,timeout):
            if self.position>=len(raw):
                cap._stop.set();raise queue.Empty
            start=time.perf_counter()
            end=min(self.position+192000,len(raw))
            values=raw[self.position:end].astype(np.float32)
            self.position=end
            input_end[0]=end/1920000
            if hasattr(self,'before'):timings.append(start-self.before)
            self.before=start
            return ((values[::2]-127.5)+1j*(values[1::2]-127.5))/128
    cap._queue=Input()
    start=time.perf_counter();cap._convert();elapsed=time.perf_counter()-start
    tag=f'{args.label}-rtl{args.serial}-join{args.join:g}'
    np.savez(path/f'{tag}.npz',latents=np.asarray(received),weights=np.asarray(weights))
    if args.pcm and pcm: np.save(path/f'{tag}.pcm.npy',np.concatenate(pcm))
    report=dict(name=args.name,serial=args.serial,label=args.label,join_s=args.join,manual=args.manual,
                input_duration_s=len(raw)/1920000,processing_s=elapsed,
                service_p95_s=float(np.percentile(timings,95)),service_max_s=float(max(timings)),
                decoded_gops=len(rows),rows=rows,events=events)
    save(path/f'{tag}.json',report)
    print(args.name,args.serial,args.label,args.join,'GOPs',len(rows),'first',rows[:1],
          'processing',round(elapsed,2),'seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['capture','replay']);p.add_argument('name')
    p.add_argument('--root', type=Path, required=True);p.add_argument('--source',type=Path);p.add_argument('--latents',type=Path)
    p.add_argument('--provenance',default='Operator-supplied source; verify source separation before making quality claims')
    p.add_argument('--mode',default='video',choices=['video','analog_av']);p.add_argument('--tx',type=float,default=-10)
    p.add_argument('--rx',type=float,default=37.2);p.add_argument('--serial',default='1001')
    p.add_argument('--join',type=float,default=0);p.add_argument('--label',default='baseline')
    p.add_argument('--manual',action='store_true');p.add_argument('--correction',type=float,default=0)
    p.add_argument('--pcm',action='store_true');args=p.parse_args()
    ROOT=args.root.resolve();SOURCE=args.source;LATENTS=args.latents
    (capture if args.action=='capture' else replay)(args)
