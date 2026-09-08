"""Verified HTTPS using the host's certificate store in portable Linux builds."""

import os
from pathlib import Path
import ssl
import sys


_SYSTEM_CA_FILES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
)


def download_ssl_context() -> ssl.SSLContext:
    """Keep TLS verification enabled when bundled OpenSSL's CA path is absent."""
    context = ssl.create_default_context()
    # PyInstaller carries OpenSSL's build-time paths across distributions, e.g.
    # /usr/lib/ssl on Ubuntu does not exist on Arch/Manjaro. Use a host-provided
    # bundle only when defaults loaded no roots. Explicit operator overrides
    # (including empty values) retain their normal OpenSSL semantics.
    if (
        sys.platform.startswith("linux")
        and not any(name in os.environ for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"))
        and context.cert_store_stats()["x509_ca"] == 0
    ):
        for filename in _SYSTEM_CA_FILES:
            if Path(filename).is_file():
                context.load_verify_locations(cafile=filename)
                break
    return context
