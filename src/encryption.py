"""
AES-256 Field-Level Encryption.
Encrypts sensitive applicant data before storage.
Key is derived from ENCRYPTION_KEY env var using PBKDF2.
Falls back to deterministic masking if key not set (for demo/dev).
"""
import os
import base64
import hashlib

_KEY_ENV = os.getenv("ENCRYPTION_KEY", "")

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


def _derive_key(password: str, salt: bytes = b"credisense_salt_v1") -> bytes:
    if not CRYPTO_AVAILABLE:
        return b""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100_000)
    return kdf.derive(password.encode())


_AES_KEY = _derive_key(_KEY_ENV) if (_KEY_ENV and CRYPTO_AVAILABLE) else None


def encrypt_field(plaintext: str) -> str:
    """Encrypt a string field. Returns base64-encoded ciphertext."""
    if not _AES_KEY or not CRYPTO_AVAILABLE:
        return "[ENC:" + hashlib.sha256(plaintext.encode()).hexdigest()[:16] + "]"
    aesgcm = AESGCM(_AES_KEY)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def decrypt_field(ciphertext: str) -> str:
    if not _AES_KEY or not CRYPTO_AVAILABLE:
        return ciphertext
    if ciphertext.startswith("[ENC:"):
        return ciphertext
    try:
        raw = base64.b64decode(ciphertext.encode())
        nonce, ct = raw[:12], raw[12:]
        return AESGCM(_AES_KEY).decrypt(nonce, ct, None).decode()
    except Exception:
        return "[DECRYPTION_ERROR]"


def mask_pii_strict(value, field_type: str) -> str:
    """Strict PII masking — buckets or hashes, never raw values."""
    def _bucket(v, thresholds, labels):
        for i, t in enumerate(thresholds[1:]):
            if float(v) < t:
                return labels[i]
        return labels[-1]

    masks = {
        "income_lpa": lambda v: f"INR_{_bucket(v,[0,3,5,10,20,50],['<3L','3-5L','5-10L','10-20L','20-50L','>50L'])}",
        "age":        lambda v: f"AGE_{_bucket(v,[0,25,35,45,55,100],['<25','25-35','35-45','45-55','>55'])}",
        "experience": lambda v: f"EXP_{_bucket(v,[0,2,5,10,20,100],['<2yr','2-5yr','5-10yr','10-20yr','>20yr'])}",
        "name":       lambda v: f"USR_{hashlib.sha256(str(v).encode()).hexdigest()[:8]}",
        "phone":      lambda v: f"PH_***{str(v)[-4:]}",
        "email":      lambda v: f"{str(v)[0]}***@{str(v).split('@')[-1] if '@' in str(v) else '***'}",
        "pan":        lambda v: f"PAN_{hashlib.sha256(str(v).encode()).hexdigest()[:8].upper()}",
        "aadhar":     lambda v: f"UID_****{str(v)[-4:]}",
    }
    fn = masks.get(field_type)
    if fn:
        try:
            return fn(value)
        except Exception:
            pass
    return f"[{field_type.upper()}_MASKED]"


def encryption_status() -> dict:
    return {
        "encryption_available": CRYPTO_AVAILABLE,
        "key_configured": bool(_KEY_ENV),
        "mode": "AES-256-GCM" if (_AES_KEY and CRYPTO_AVAILABLE) else "SHA-256 hash fallback",
    }
