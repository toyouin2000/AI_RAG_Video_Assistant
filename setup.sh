#!/bin/bash
set -e

echo "Installing Deno..."

curl -fsSL https://deno.land/install.sh | sh

export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"

echo "Deno installed at:"
which deno

deno --version
