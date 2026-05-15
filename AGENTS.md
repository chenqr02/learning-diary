# Project Rules

## Hugo Math Rendering

This Hugo site uses KaTeX auto-render, while Goldmark parses Markdown before KaTeX runs. To avoid Markdown corrupting formulas, every block/display formula in content files must be wrapped with the existing `rawhtml` shortcode.

Use this format for display math:

```md
{{< rawhtml >}}
$$
\mathcal{L} = \sum_i x_i
$$
{{< /rawhtml >}}
```

Do not write bare display math directly:

```md
$$
\mathcal{L} = \sum_i x_i
$$
```

Reasons:

- `_` inside formulas can be parsed as Markdown emphasis before KaTeX sees it.
- `<t>` or similar token notation can be parsed as HTML.
- A standalone `=` line inside a formula can be parsed as a Markdown setext heading underline.

For multi-line equations, prefer `aligned` and avoid a bare line containing only `=`:

```md
{{< rawhtml >}}
$$
\begin{aligned}
\mathcal{L}_{SDFT}
&= \mathcal{L}_{SFT} + \lambda \mathcal{L}_{SD}
\end{aligned}
$$
{{< /rawhtml >}}
```

Inline math like `$x_t$` is acceptable for simple formulas. If inline math is complex or contains risky Markdown/HTML characters, convert it to a wrapped display formula.

After adding or editing math-heavy content, run:

```sh
hugo
rg -n '<p>\$\$|\$\$</p>|<em>|id="mathcal|id="nabla|id="frac|id="sum|id="pi|id="-log' public/notes public/mianshi
```

The second command should produce no output.
