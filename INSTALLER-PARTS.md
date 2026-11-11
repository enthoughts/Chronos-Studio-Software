# Standalone installer (split parts)

`chronos-studio-standalone_0.1.0_amd64.deb` is 103.97 MiB, above GitHub's 100 MiB per-file limit, so it is stored as:

- `chronos-studio-standalone_0.1.0_amd64.deb.part1`
- `chronos-studio-standalone_0.1.0_amd64.deb.part2`

Rebuild it on your machine:

```bash
./rebuild-deb.sh
sudo apt install --reinstall ./chronos-studio-standalone_0.1.0_amd64.deb
```

The script concatenates the parts and verifies SHA-256 against `chronos-studio-standalone_0.1.0_amd64.deb.sha256`.
