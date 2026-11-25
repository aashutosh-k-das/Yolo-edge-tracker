#!/usr/bin/env bash
# Install / remove the boot service.   bash deploy/service/install_service.sh [--uninstall]
set -euo pipefail
DIR=$(cd "$(dirname "$0")/../.." && pwd)
UNIT=/etc/systemd/system/yolo-tracker.service

if [ "${1:-}" = "--uninstall" ]; then
    sudo systemctl disable --now yolo-tracker || true
    sudo rm -f "$UNIT" && sudo systemctl daemon-reload
    echo "Removed."; exit 0
fi

sed -e "s#@USER@#$USER#g" -e "s#@DIR@#$DIR#g" "$DIR/deploy/service/yolo-tracker.service" | sudo tee "$UNIT" >/dev/null
chmod +x "$DIR/deploy/service/start_tracker.sh"
sudo systemctl daemon-reload
sudo systemctl enable --now yolo-tracker
sleep 3
systemctl --no-pager status yolo-tracker | head -n 12
cat <<MSG

Installed. Useful commands:
  journalctl -u yolo-tracker -f          # live logs
  sudo systemctl restart yolo-tracker    # after editing deploy/service/tracker.env
  sudo systemctl stop yolo-tracker       # stop (e.g. to free the camera for testing)
  bash deploy/service/install_service.sh --uninstall
Stream (if --stream-port 8080): http://$(hostname -I | awk '{print $1}'):8080
MSG
