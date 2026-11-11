"""Pytest defaults ensuring zero-config test execution without external .env credentials."""

import os

TEST_ENV_DEFAULTS = {
    "ENVIRONMENT": "test",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "REDIS_URL": "",
    "WORKER_SECRET": "test-worker-secret",
    "R2_ACCOUNT_ID": "test-r2-account",
    "R2_ACCESS_KEY_ID": "test-r2-key",
    "R2_SECRET_ACCESS_KEY": "test-r2-secret",
    "R2_PUBLIC_URL": "https://cdn.example.com",
    "GROQ_API_KEY": "",
    "HF_TOKEN": "",
    "OPENAI_API_KEY": "",
}

for _k, _v in TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
