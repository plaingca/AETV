"""Portable builds discover host roots without weakening TLS verification."""

import ssl

import pytest

import aetv.tls as tls


@pytest.fixture
def empty_linux_context(monkeypatch):
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(tls.ssl, "create_default_context", lambda: context)
    monkeypatch.setattr(tls.sys, "platform", "linux")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    return context


@pytest.fixture
def host_ca_file(tmp_path):
    cert = ssl.create_default_context().get_ca_certs(binary_form=True)[0]
    path = tmp_path / "host-ca.pem"
    path.write_text(ssl.DER_cert_to_PEM_cert(cert), encoding="ascii")
    return str(path)


def test_linux_loads_host_bundle_when_bundled_openssl_has_no_roots(
    host_ca_file, empty_linux_context, monkeypatch, tmp_path,
):
    monkeypatch.setattr(tls, "_SYSTEM_CA_FILES", (str(tmp_path / "absent.pem"), host_ca_file))
    context = tls.download_ssl_context()
    assert context is empty_linux_context
    assert context.cert_store_stats()["x509_ca"] == 1
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname


@pytest.mark.parametrize("name,value", [
    ("SSL_CERT_FILE", "/custom/roots.pem"), ("SSL_CERT_FILE", ""),
    ("SSL_CERT_DIR", "/custom/certs"), ("SSL_CERT_DIR", ""),
])
def test_explicit_trust_configuration_is_not_replaced(
    host_ca_file, empty_linux_context, monkeypatch, name, value,
):
    monkeypatch.setattr(tls, "_SYSTEM_CA_FILES", (host_ca_file,))
    monkeypatch.setenv(name, value)
    assert tls.download_ssl_context().cert_store_stats()["x509_ca"] == 0


def test_existing_trust_store_is_preserved(host_ca_file, empty_linux_context, monkeypatch, tmp_path):
    empty_linux_context.load_verify_locations(cafile=host_ca_file)
    invalid = tmp_path / "invalid.pem"
    invalid.write_text("not a certificate")
    monkeypatch.setattr(tls, "_SYSTEM_CA_FILES", (str(invalid),))
    assert tls.download_ssl_context() is empty_linux_context


def test_missing_roots_still_require_verification(empty_linux_context, monkeypatch):
    monkeypatch.setattr(tls, "_SYSTEM_CA_FILES", ())
    context = tls.download_ssl_context()
    assert context.cert_store_stats()["x509_ca"] == 0
    assert context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname


def test_non_linux_keeps_platform_defaults(host_ca_file, empty_linux_context, monkeypatch):
    monkeypatch.setattr(tls, "_SYSTEM_CA_FILES", (host_ca_file,))
    monkeypatch.setattr(tls.sys, "platform", "win32")
    assert tls.download_ssl_context().cert_store_stats()["x509_ca"] == 0
