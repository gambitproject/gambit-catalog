# Gambit catalog of games

A curated collection of games from the game theory literature, part of the
[Gambit project](https://www.gambit-project.org).

Each game has a canonical name (its *slug*, e.g. `books/myerson1991/fig2_1`) and is stored
in Gambit's file formats: `.efg` for extensive games and `.nfg` for strategic games, with
`.ef` drawings for extensive games. `games/manifest.json` indexes every game with its title,
description and structural statistics.

The catalog is a dataset, versioned and released independently of Gambit. A game is
identified by its slug together with a catalog release (or commit).

- Browse the catalog: <https://gambit-catalog.readthedocs.io/>
- Load a game in Python with [pygambit](https://gambitproject.readthedocs.io/):
  `pygambit.catalog.load("books/myerson1991/fig2_1")`
- Contribute a game: see [doc/developer.rst](doc/developer.rst)

## Licence

The games and their metadata are licensed under CC BY 4.0; the tooling under
GPL-2.0-or-later. See [LICENSE](LICENSE).
