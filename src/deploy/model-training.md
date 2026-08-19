# Model-Training Service Deployment

The live service must run with `--enable-operator-console`; its existing
operator token authorizes native still capture. The training service uses a
separate browser/API token.

1. Replace the three token placeholders in
   `person-recognition-model-training.service`.
2. Install the unit in `/etc/systemd/system/`.
3. Add `model-training-nginx.conf` to the existing HTTPS server block.
4. Reload nginx and enable the service.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now person-recognition-model-training.service
sudo nginx -t
sudo systemctl reload nginx
```

Open `/model-training/`, enter the model-training token, and press `Connect`.
The first connection refreshes the product catalog when the local cache is
empty. No additional Python packages are required beyond `requirements.txt`.
