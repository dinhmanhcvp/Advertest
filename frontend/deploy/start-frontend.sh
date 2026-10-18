#!/bin/sh
set -eu
exec npm run start -- -p "${PORT:-3000}"
