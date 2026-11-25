{
  description = "Reachy Mini conversation app development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              just
              python
              uv
              # pycairo and pygobject (via reachy-mini) build from source against these.
              cairo
              glib
              gobject-introspection
              pkg-config
            ];

            UV_PYTHON = "${python}/bin/python3.12";
            UV_PYTHON_DOWNLOADS = "never";

            # Prebuilt manylinux wheels such as numpy load libstdc++ at runtime.
            LD_LIBRARY_PATH = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux (
              pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]
            );
          };
        });
    };
}
