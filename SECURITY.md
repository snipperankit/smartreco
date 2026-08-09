# Security Policy

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability or exposed secret.
Use the repository's **Security > Report a vulnerability** private reporting
flow. Include reproduction steps, affected endpoints, and impact when known.

Secrets are configured only through GitHub Actions or Render environment
variables. If a secret may have been exposed, revoke it before investigating
or deploying a fix.
