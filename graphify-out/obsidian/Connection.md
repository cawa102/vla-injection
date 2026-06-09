---
source_file: "docs/gpu/Connection.md"
type: "document"
community: "GPU Runbook & Kelvin2"
tags:
  - graphify/document
  - graphify/EXTRACTED
  - community/GPU_Runbook__Kelvin2
---

# Connection.md

## Connections
- [[Kelvin2 — Connecting (macOS)]] - `defined_in` [EXTRACTED]

#graphify/document #graphify/EXTRACTED #community/GPU_Runbook__Kelvin2

## 📄 Source

`docs/gpu/Connection.md`

# Kelvin2 — Connecting (macOS)

> **Source:** <https://ni-hpc.github.io/nihpc-documentation/Connecting%20to%20Kelvin2/> — fetched 2026-06-03.
> Tailored to **macOS + QUB network access**. Commands run in **Terminal.app** / iTerm2.

## 0. Get an account first

Apply at <https://www.ni-hpc.ac.uk/Kelvin2/Access/>. Allow **~48 h**; credentials arrive by email.
- **Username** = your QUB **staff / student number**.
- **Password** = your **QUB Active Directory** password (same as QUB email/portal).

## 1. Which address — depends on where you are

| Situation | Host | Port | Auth |
|-----------|------|------|------|
| **On QUB network** (campus wifi/ethernet **or QUB VPN**) | `kelvin2.qub.ac.uk` | **22** | QUB AD password + MFA |
| **Off QUB network** (no VPN) | `login.kelvin.alces.network` | **55890** | **SSH key** + MFA |

Since you have **QUB network access**, the simplest path is: connect to the **QUB VPN** (or be on campus), then
use `kelvin2.qub.ac.uk:22` with your password. The SSH-key route is the fallback when you're off-network and
can't VPN.

> ⚠️ **MFA is mandatory** on all Kelvin2 accounts now (see §4). You set it up once on first login.

## 2. Connect from QUB network (password) — primary path

```bash
ssh <studentnumber>@kelvin2.qub.ac.uk
```

You land on one of four login nodes (`login1`–`login4`); prompt looks like
`[<studentnumber>@login1 [kelvin2] ~]$`. **Login nodes are for light work only** — editing, transfers, job
submission/monitoring. Never run training/inference there (see [`Start.md`](./Start.md)).

## 3. Connect from off-network (SSH key) — fallback

**Generate a key on the Mac** (do this once):

```bash
ssh-keygen -t rsa -f ~/.ssh/kelvin2-key      # you MUST set a passphrase when prompted
cat ~/.ssh/kelvin2-key.pub                    # copy this public key
```

**Install the public key on Kelvin2.** Easiest while you still have on-network access — append it to
`~/.ssh/authorized_keys` on the cluster:

```bash
# on Kelvin2 (after a password login from QUB network):
vi ~/.ssh/authorized_keys
# Shift+G (end of file) → Shift+A → Enter → paste the .pub line → Esc → :wq
```

If you have **no** way onto the network, email the public key via <https://www.ni-hpc.ac.uk/contact/> for an
admin to install it.

**Then connect from anywhere:**

```bash
ssh -p 55890 -i ~/.ssh/kelvin2-key <studentnumber>@login.kelvin.alces.network
```

## 4. Multi-factor authentication (one-time setup)

On your first successful login, generate the MFA secret on the cluster:

```bash
/opt/flight/bin/flight mfa generate
```

Scan the QR code with an authenticator app (Microsoft Authenticator, Google Authenticator, etc.). On subsequent
logins you'll be prompted for the 6-digit code. **Lost device → reset via** <https://www.ni-hpc.ac.uk/contact/>.

## 5. Recommended `~/.ssh/config` (macOS convenience)

> Standard SSH client config — **not** from the official docs, but it saves typing the long off-network command
> and lets `scp`/`rsync` reuse the alias. Edit `~/.ssh/config`:

```sshconfig
# On QUB network / VPN
Host kelvin2
    HostName kelvin2.qub.ac.uk
    User <studentnumber>
    Port 22

# Off network (alces login)
Host kelvin2-ext
    HostName login.kelvin.alces.network
    User <studentnumber>
    Port 55890
    IdentityFile ~/.ssh/kelvin2-key

# Data-mover node for large transfers (see Start.md)
Host kelvin2-dm
    HostName dm1.kelvin.alces.network
    User <studentnumber>
    Port 55890
    IdentityFile ~/.ssh/kelvin2-key
```

Then simply: `ssh kelvin2` (on-network) or `ssh kelvin2-ext` (off-network).

## 6. Host-key warnings on reconnect

There are four login nodes behind `kelvin2.qub.ac.uk`, so SSH may warn about a changed host key. To pre-trust
them, append the official fingerprints (IPs `143.117.27.19–22`, `.51`) to `~/.ssh/known_hosts`. The published
ecdsa key for those hosts is:

```
kelvin2.qub.ac.uk,143.117.27.19 ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBOpr/Rr+2UUve4tQPVnpEc383LCNG4El2hgmnmgN15aDm5XpE3l6qjJ4fpiOaVe386bU+79FPnG1HURvulmZocU=
```

(repeat the same key line for `.20`, `.21`, `.22`, `.51`). Only do this if the warning is the expected
multi-login-node case, **not** if you suspect an actual MitM.

## 7. File transfer — quick reference (full detail in [`Start.md`](./Start.md))

```bash
# On QUB network:
scp ./file.txt <studentnumber>@kelvin2.qub.ac.uk:/mnt/scratch2/users/<studentnumber>/

# Off network:
scp -P 55890 -i ~/.ssh/kelvin2-key ./file.txt \
    <studentnumber>@login.kelvin.alces.network:/mnt/scratch2/users/<studentnumber>/
```

Use the **data-mover nodes** `dm1.kelvin.alces.network` / `dm2.kelvin.alces.network` for large transfers
(checkpoints, datasets) rather than the login nodes.

## Troubleshooting

- **Permission denied (publickey)** off-network → key not in `authorized_keys`, or wrong `-i` path / port.
- **Connection refused on :22 off-network** → you're not on QUB network; use the alces host on `:55890`, or
  connect QUB VPN first.
- **Repeated MFA prompts during transfers** → reuse a single SSH connection (`ControlMaster`) or transfer via a
  data-mover node.

