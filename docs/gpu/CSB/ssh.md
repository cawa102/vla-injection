# CSB `ecs3-0202` — Connect & Resume Runbook (get a shell on the box)

> **What works (2026-06-16):** the reliable access path is the **VS Code Remote Tunnel (`code tunnel`)** — an
> **outbound** connection from the box. **Direct inbound SSH does NOT currently work** (the box's SSH is on port
> 2222, but inbound SSH from off-box is dropped by a campus/host firewall — see §3). The tunnel is **IP-independent**,
> so the box's changing DHCP address does not matter for it. Box facts: [`pc-spec.md`](./pc-spec.md).

---

## 0. TL;DR — 次回の再開手順(これだけ)

1. **箱にログイン**(物理席 or 画面共有)。IPは見なくてよい(tunnel はIP非依存)。
2. 箱でトンネル起動(サービス常駐済みなら不要 → §2.3):
   ```bash
   ~/code tunnel --accept-server-license-terms
   ```
3. **Mac の VS Code** → `⌘+Shift+P` → `Remote Tunnels: Connect to Tunnel` → **同じ GitHub** でサインイン → `ecs3` を選択。
4. VS Code 左下が緑+`ecs3` になったら **Terminal → New Terminal** = 箱のシェル。
5. 作業開始(bring-up = [`plan.md`](./plan.md))。

> ⚠️ ブラウザ版 `https://vscode.dev/tunnel/ecs3` が**真っ暗**なときは、Safari をやめ Chrome/Edge にするか、上の **VS Code アプリ**を使う(同じ GitHub アカウントであることが必須)。

---

## 1. The box (facts established this session)

| 項目 | 値 |
|------|----|
| Host / FQDN | `ecs3-0202` / `ecs3-0202.eeecs.qub.ac.uk` |
| Login user | `40473058@eeecs.qub.ac.uk` (UPN形式 — `@` を含む) |
| IP (例) | `143.117.95.122` — **DHCPで起動毎に変わりうる**。DNSが遅延して別IP(`.55`)を返したこともある → IPは箱で実測する |
| sshd port | **2222**(標準の22ではない) |
| sudo | **無し**(サーバ側のFW/sshd設定は変更不可) |
| GPU / OS | 2× RTX A5000 24GB / Ubuntu 24.04 / driver 595, CUDA 13.2 |

---

## 2. Primary path — VS Code Remote Tunnel(working, sudo不要, IP非依存)

### 2.1 IPの確認(任意 / 直SSHを試すときだけ必要。tunnelには不要)
箱で:
```bash
hostname -I
ip -4 addr show | grep inet
```

### 2.2 One-time setup(箱で一度だけ)
```bash
cd ~
wget -qO vscode_cli.tar.gz "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64"
tar xf vscode_cli.tar.gz            # → ~/code (static binary)
./code tunnel --accept-server-license-terms
```
- 表示される `https://github.com/login/device` を箱の firefox で開き、**コード**を入力して GitHub 認証。
- マシン名を聞かれたら `ecs3`(既定でも可)。
- `➜ Open: https://vscode.dev/tunnel/ecs3` が出れば稼働中。**この端末は開いたまま**にする(閉じると切れる)。

### 2.3 常駐させる(推奨 / sudo不要 → 次回は箱で何もしなくてよくなる)
```bash
~/code tunnel service install --accept-server-license-terms
# 状態確認 / ログ / 停止:
~/code tunnel service log
~/code tunnel service uninstall
```
> 注意: これは systemd **--user** サービス。**ログアウト後も維持**するには linger が必要:
> `loginctl enable-linger "$USER"`(権限が無ければ拒否される)。共有機なので、最悪は §0 の手動起動でよい。

### 2.4 Mac から接続(毎回)
- **VS Code アプリ**(推奨): `⌘+Shift+P` → `Remote Tunnels: Connect to Tunnel` → 同じ GitHub → `ecs3`。
  - 拡張が無ければ「**Remote - Tunnels**」をインストール。
  - 接続後 **Terminal → New Terminal**(`Ctrl+\``)で箱のシェル。`File → Open Folder` で repo を開く。
- **ブラウザ版**(代替): Chrome/Edge で `https://vscode.dev/tunnel/ecs3`(同じ GitHub でサインイン)。

---

## 3. Secondary path — direct SSH(現状ブロック / 開通したとき用に記録)

Mac 側のSSH鍵・configは設定済みだが、**inbound SSH は通らない**(下記の遮断)。将来 FW が開いたら使える。

### 3.1 Mac 側設定(設定済み)
- `~/.ssh/id_ed25519`(ed25519鍵, パスフレーズ無し)。公開鍵は箱の `~/.ssh/authorized_keys` に登録済み。
- `~/.ssh/config`:
  ```sshconfig
  Host ecs3
      HostName ecs3-0202.eeecs.qub.ac.uk
      Port 2222
      User 40473058@eeecs.qub.ac.uk
      IdentityFile ~/.ssh/id_ed25519
      StrictHostKeyChecking accept-new
      ServerAliveInterval 30
      ServerAliveCountMax 4
  ```
  > DHCPでIPが変わるので `HostName` はFQDN推奨。FQDNのDNSが古い場合のみ、§2.1で得た実IPを一時的に `HostName` に入れる。

### 3.2 到達性テスト
```bash
nc -z -G 6 -v ecs3-0202.eeecs.qub.ac.uk 2222     # TCP到達(これは成功する)
ssh -o BatchMode=yes -o ConnectTimeout=10 ecs3 'echo OK; hostname'
```

### 3.3 既知の遮断(2026-06-16 時点)
- **症状**: `nc` でTCP(2222)は **succeeded** なのに、`ssh` は `kex_exchange_identification: read: Operation timed out` /
  `banner exchange ... timed out`。**最初は応答(Connection closed)→ 数回試行後に無応答(timeout)に変化**。
- **原因(推定)**: ネットワークのIPS/FW(またはホスト側フィルタ)が、繰り返しのSSH試行後に Mac→箱:2222 の **SSHを落とす**。
  `UseDNS` 遅延は否定済み(75秒待っても banner が一切来ない)。fail2ban-client は未インストール。
- **対処**: **sudo が無いので箱からは解除不可**。**連打しない**(無応答が続くと遮断が延びる)。試すなら**1回だけ**。開通の見込みが要るなら EEECS IT に「ecs3-0202 の 2222 への inbound SSH 許可」を依頼。
  → 当面は **§2 の tunnel が正規ルート**。

---

## 4. 接続後の環境チェック(GO/NO-GO)

箱のシェル(tunnel経由)で:
```bash
echo "== HOME =="; df -h ~ | tail -1
echo "== net =="; timeout 10 wget -q -O/dev/null https://repo.anaconda.com/miniconda/ && echo anaconda_OK || echo anaconda_NG
timeout 10 wget -q -O/dev/null https://huggingface.co && echo hf_OK || echo hf_NG
echo "== GPU =="; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "== toolchain =="; python3 --version; for t in conda git pip3 nvcc; do printf "%s: " $t; command -v "$t" || echo "なし"; done
echo "== cpu/ram =="; echo "$(nproc) cores"; free -h | awk '/Mem/{print $2" RAM"}'
```
- 期待: GPU=`RTX A5000, 24564 MiB ×2`。**`conda` も `pip` も無く `uv` だけが有る(2026-06-17 箱で確認)** → 環境構築は **uv** で行う(§5)。
- `wget` で `~/code` を落とせている時点で**外向きHTTPは到達可**(anaconda/HF の個別到達は上で確認)。

---

## 5. 次へ — bring-up(初回のみ）

> **箱の事実(2026-06-17、実機確認):** `git`/`conda`/`pip` は無く `uv` だけが有る・sudo 無し →
> git は **micromamba(単体静的バイナリ)で別途導入**(下記0a)、env は **uv** で作る(uv が Python 3.10・venv・依存を全部管理、root不要)。
> `uv.lock` は repo にコミット済 → `uv sync` で model-free 依存を**厳密再現**できる。

環境OKなら(root不要):
```bash
# 0a) git が無ければ micromamba(単体静的バイナリ)で導入 — この箱は git/conda/pip 無し・sudo無し
cd ~
wget -O ~/micromamba https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
chmod +x ~/micromamba
export MAMBA_ROOT_PREFIX="$HOME/.mamba"
~/micromamba create -y -p "$HOME/git-env" -c conda-forge git
export PATH="$HOME/git-env/bin:$PATH"
echo 'export PATH="$HOME/git-env/bin:$PATH"' >> ~/.bashrc   # 恒久化
git --version

# 0b) uv に Python 3.10 を用意させる(システムに無くてOK)
uv python install 3.10

# 1) repo は必ず git clone(ZIP等のDLは .git が無く capture_env の commit=None → 再現性NG)
#    private repo: 認証を聞かれたら PAT(Contents:Read) を入力。共有機ゆえ credential は保存しない
git clone https://github.com/cawa102/vla-injection.git ~/vla-injection
cd ~/vla-injection

# --- bring-up step 2 ゲート: model-free 395 テスト(torch 不要・repo を先に検証)---
uv venv --python 3.10          # .venv を 3.10 で作成(OpenVLA スタックに合わせる)
uv sync                        # uv.lock から厳密再現(core deps + dev: pytest/ruff)
uv run pytest -q               # 期待: 395 passed(Mac とパリティ)

# --- bring-up step 1 ゲート: CUDA(torch を同じ venv に追加)---
uv pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # True / RTX A5000
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

> ⚠️ **`uv sync` を torch インストール後に再実行しない** — lock に無い torch を削除する。再同期が要るなら
> `uv sync --inexact`(既存パッケージを残す)。`uv run` が `.venv` を使うので `activate` は不要。

以降は [`plan.md`](./plan.md) の bring-up ラダー(OpenVLA bf16 → LIBERO → L2 → GCG/D8)。

---

## 6. Gotchas(今回の教訓)

- **SSH は 2222**(22 は塞がれている)。
- **DNS が古いことがある**(`.55` を返した。実体は `.122`)→ IPは `hostname -I` で実測。
- **username に `@`** → `ssh -l '40473058@eeecs.qub.ac.uk' …` のように引用。
- **sudo 無し** → サーバ側(fail2ban/UseDNS/FW)は触れない。
- **inbound SSH を連打しない**(IPS/fail2ban風の遮断を延ばす)。
- **tunnel 端末は開いたまま**(または §2.3 でサービス化)。
- **ブラウザ版が真っ暗** → VS Code アプリ + 同じ GitHub、Chrome/Edge を使う。
- **`git` が無い**(conda/pip も無い・sudo 無し) → **micromamba 単体バイナリ**で conda-forge `git` をユーザ領域へ(§5 0a)。tarball パイプ(`wget -qO- … | tar -xvj`)は途中で切れて失敗したので、**単体バイナリを直接DL**する。
- **repo は必ず `git clone`** — ZIP/DL では `.git` が無く `capture_env()` の `git_commit=None` → `test_git_commit_is_resolved_in_this_repo` が落ち、登録ランの commit 記録も取れない(再現性NG)。
