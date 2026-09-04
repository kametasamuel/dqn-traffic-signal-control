Place the SUMO-RL bundled network/route files here:
- single-intersection.net.xml
- single-intersection-vhvh.rou.xml
- single-intersection-horizontal.rou.xml
- single-intersection-vertical.rou.xml
- single-intersection-gen.rou.xml (held-out evaluation route)

These ship with the sumo-rl package under its `nets/` directory — copy them
in rather than redefining the network, so the intersection topology matches
the proposal exactly.
