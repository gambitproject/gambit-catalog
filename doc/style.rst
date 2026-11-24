.. _catalog-style:

Style guide for catalog entries
===============================

This guide covers what a contributor writes for a catalog entry and what the build
cannot check: the source, the name, and the description.
Each rule is written so that it can be checked by reading the entry.
The checklist at the end gathers them for reviewing a contribution.

The build checks the rest: every entry has a ``source.yaml``; every key in it is in
``references.bib``; and the title in the game file is the one derived from ``source.yaml``.

Source
------

``source.yaml`` sits in the entry's directory, beside the game file::

   source:
     key: Sel75
     locator: Figure 1
   origin:
     key: Lie60
     locator: Table 2
     note: payoffs rescaled
   name: Selten's horse

``source``
   The work the game file was transcribed from: the book, article or chapter in which a
   reader finds this game, drawn or tabulated as in the file.
   It is required.

   - ``key`` is the work's key in ``references.bib``. Add an entry there if there is none.
   - ``locator`` says where in the work the game is, written as the work labels it, with
     the label spelled out: ``Figure 2``, ``Figure 1(a)``, ``Table 3``, ``Exercise 29.6``,
     ``Section 4``, ``Example 2``. Use the work's own numbering, including chapter numbers
     (``Figure 5.10``). Omit it only when the work presents a single game and gives it no
     label.

``origin``
   The earlier work the game itself comes from, when the source took it from elsewhere.
   Give it only when the source presents a game first published in another work, such as
   an experiment run on a game from an earlier paper. At most one.

   - ``key`` and ``locator`` as for ``source``.
   - ``note`` says, in a few words, how the entry differs from the game in the origin, if it
     does: ``payoffs rescaled``, ``payoffs made much smaller``, ``one move removed``.

   A work that only supplied a story or framing for the game, and a related game in
   another work, are not an origin: mention them in the description.

``name``
   The game's conventional name, when it has one: a name that the literature uses for this
   game, or that the source itself gives it (``Selten's horse``, ``Centipede game``,
   ``Battle of the Sexes``, ``Stripped-down poker``). Most games have none; omit the field
   rather than invent one. A description of what the game shows ("A game with imperfect
   recall") is not a name.

   Write it as the literature does, including its capitals (``Prisoner's Dilemma``), with no
   leading article and no citation.

Title
-----

Do not write the title by hand. It is derived from ``source.yaml`` as::

   <citation>, <locator>[: <name>]

for example ``Selten (1975), Figure 1: Selten's horse`` or
``Jakobsen et al. (2016), Figure 1(a)``. After editing ``source.yaml``, run
``python build.py --sync-titles`` to rewrite the title in the game file.

Description
-----------

The description is the comment string of the game file. It is shown wherever the game
is listed, beside the title, so it does not repeat what the title already says.

- **Open with what the game is or shows.** The first sentence says what kind of game this
  is, or what the source uses it to illustrate: "A counterexample to show that extensive
  and normal form concepts of perfectness do not coincide." Start with the substance, not
  with "This is" or "This game".
- **Do not restate the source.** No "Figure 2 from ...", "An example from ...",
  "presented in ``:cite:p:`...```": the title gives the source. Cite other works freely: an
  origin, the source of a framing or story, a related game.
- **Cite with** ``:cite:p:`key```, for a key in ``references.bib``.
- **Say what the file fixes that the source leaves open.** If the source has parameters,
  say which values the file uses ("The probability that the follower observes the
  leader's action is 0.99."). If the file departs from the source in any way (corrected
  payoffs, relabelled players, a move removed), say so.
- **Explain labels the reader needs.** When the description refers to players, actions
  or strategies, use the labels in the file.
- **Be specific.** Name the property the game illustrates and, where the source gives
  them, the facts that show it: the number of equilibria, which profile is or is not an
  equilibrium, which refinement separates them. Avoid descriptions that would fit any
  game of its kind ("An example of an imperfect-information game") when the source says
  more.
- **Format.** Plain text, in paragraphs separated by a blank line; lines may be wrapped
  anywhere within a paragraph. Use ``:math:`...``` for mathematics. A ``See also`` block
  names a related entry's citation followed by an indented line saying how it differs.
  American spelling (behavior, signaling), as in Gambit's documentation.

Checklist
---------

For each entry:

1. ``source.key`` names the work the file was transcribed from, not a work that cites it.
2. ``source.locator`` is the work's own label, spelled out, and matches the drawing or
   table in the work.
3. ``origin`` is present exactly when the game was first published elsewhere; its
   ``note`` describes any difference.
4. ``name`` is present only for a conventional name, written as the literature writes it.
5. The title has been regenerated with ``--sync-titles``.
6. The description's first sentence says what the game is or shows, without "This is".
7. The description does not restate the source or its locator.
8. Parameter values and departures from the source are stated.
9. Every label the description mentions appears in the file.
10. Citations use ``:cite:p:`` with keys in ``references.bib``.
