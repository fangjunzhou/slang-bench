{
  description = "A Slang benchmark report";

  inputs =
    {
      nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
      flake-utils.url = "github:numtide/flake-utils";
    };

  outputs = { self, nixpkgs, flake-utils }:
    with flake-utils.lib;
    eachSystem [
      system.x86_64-linux
      system.aarch64-darwin
    ]
      (system:
        let
          inherit (nixpkgs) lib;
          pkgs = import nixpkgs { inherit system; };
        in
        {
          devShells.default = pkgs.mkShell
            {
              buildInputs = with pkgs; [
                # Python environment.
                python3
                uv
                # LaTeX environment with comprehensive package collection
                texlive.combined.scheme-full
                # Additional tools for LaTeX workflow
                biber # Bibliography processor for biblatex
                # Code editor support (optional)
                texlab # LSP server for LaTeX
              ];
              shellHook = ''
                # Unset XCode SDK to avoid build issues on macOS
                if [[ "$(uname)" == "Darwin" ]]; then
                  export PATH="/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
                  export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
                  # Let xcrun pick the Xcode SDK; remove Nix’s SDK/toolchain overrides
                  unset SDKROOT CPATH LIBRARY_PATH NIX_CFLAGS_COMPILE NIX_LDFLAGS
                fi
                # Create the virtual environment if it doesn't exist
                if [ -d .venv ]; then
                  # Activate the virtual environment
                  source .venv/bin/activate
                  # Add .venv/bin to PATH
                  export PATH=$PWD/.venv/bin:$PATH
                else
                  echo "Environment not initialized."
                fi
              '';
            };
        }
      );
}
