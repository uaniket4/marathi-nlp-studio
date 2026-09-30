#!/bin/bash
# EC2 boot: run the Marathi NER backend container + Caddy (auto-HTTPS via nip.io).
set -x
exec > /var/log/user-data.log 2>&1

REGION=us-east-1
REG=236052539520.dkr.ecr.us-east-1.amazonaws.com
IMG=$REG/marathi-nlp-studio:latest
HOSTNAME_NIP=100.48.156.130.nip.io

# --- Docker ---
dnf update -y
dnf install -y docker
systemctl enable --now docker

# --- Pull image from ECR (instance profile provides creds) ---
for i in 1 2 3 4 5 6; do
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REG" && break
  sleep 15
done
docker pull "$IMG"

# --- Run backend, bound to localhost only (Caddy fronts it) ---
docker run -d --restart always --name backend -p 127.0.0.1:8000:8000 \
  -e QUANTIZE=1 -e TORCH_THREADS=1 -e MODEL_NAME=l3cube-pune/marathi-ner \
  -e MAX_LENGTH=256 -e CORS_ORIGINS=https://marathi-nlp-studio.vercel.app \
  "$IMG"

# --- Caddy static binary for automatic HTTPS ---
cd /usr/local/bin
curl -fsSL -o caddy.tar.gz "https://github.com/caddyserver/caddy/releases/download/v2.8.4/caddy_2.8.4_linux_amd64.tar.gz"
tar -xzf caddy.tar.gz caddy
chmod +x caddy
rm -f caddy.tar.gz

mkdir -p /etc/caddy
cat > /etc/caddy/Caddyfile <<EOF
${HOSTNAME_NIP} {
    reverse_proxy 127.0.0.1:8000
}
EOF

id caddy >/dev/null 2>&1 || useradd --system --home /var/lib/caddy --create-home --shell /usr/sbin/nologin caddy
mkdir -p /var/lib/caddy
chown -R caddy:caddy /var/lib/caddy

cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Caddy
After=network.target docker.service

[Service]
User=caddy
Group=caddy
Environment=XDG_DATA_HOME=/var/lib/caddy
Environment=XDG_CONFIG_HOME=/var/lib/caddy
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile
Restart=always
AmbientCapabilities=CAP_NET_BIND_SERVICE

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now caddy
echo "USER-DATA COMPLETE"
