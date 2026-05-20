{
  description = "Nix-managed dev environment for python-resp";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  }

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        pythonEnv = pkgs.python311.withPackages (ps: [
          ps.numpy
          ps.opencv4
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
          ];

          shellHook = ''
            echo "Entered Nix dev shell (python-resp)."
            echo "Run: python respiration_rate.py"
          '';
        };
      }
    );
}
