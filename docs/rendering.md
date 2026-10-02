# Getting formulas to render in Cursor

Cursor has three markdown renderers that accept different delimiters (checked against the renderer source and Cursor staff forum answers, September 2026).

- **Classic Markdown preview** (`Cmd+Shift+V`, or `Cmd+K` then `V` for side by side): renders `$...$` and `$$...$$` through the built-in `markdown-math` extension (KaTeX). Does not render `\(...\)` or `\[...\]`. Setting: `markdown.math.enabled` (default on). **All documents in `docs/` are written for this preview.**
- **Chat panel**: renders `\(...\)`, `\[...\]` and `$$...$$`; deliberately not `$...$`.
- **Cursor's native "Preview" tab, the WYSIWYG editor, and the plan editor**: render no math at all.
- **Jupyter notebook markdown cells**: same as the classic preview.

Rules for `$` in the classic preview: the opening `$` must not be preceded by a letter, digit, backslash or `$`, and must be followed by a non-space; the closing `$` must be preceded by a non-space and not followed by a letter or digit. Put `$$` blocks on their own lines with blank lines around them. A literal dollar sign is `\$`. Formulas inside back-ticks or code fences are shown literally. Avoid a bare `<` inside math; use `\lt` or `\le`.

Test line: if you can read $e^{j\theta} = \cos\theta + j\sin\theta$ as a typeset formula, rendering works.
