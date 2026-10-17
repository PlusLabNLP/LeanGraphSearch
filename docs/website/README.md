# LeanGraphSearch project website

Static project page for **LeanGraphSearch for Formal Theorem Proving**.

Production URL: <https://pluslabnlp.github.io/LeanGraphSearch/>

## Preview

From the repository root:

```sh
python3 -m http.server 8000 --directory docs/website
```

Open <http://localhost:8000>. No build step or third-party runtime is needed.
All fonts, figures, scripts, and styles load locally; paper links point to
[arXiv:2610.22339](https://arxiv.org/abs/2610.22339). There are no
analytics, tracking scripts, or external font requests.

## Content and figures

- The supplied anonymous submission is included, unchanged, as
  `assets/LeanGraphSearch-paper.pdf`.
- `assets/overview.png` and `assets/method.png` are cropped renderings of
  Figures 1 and 2 from that paper, excluding page headers, review line numbers,
  and surrounding text.
- `assets/cost.svg` redraws the values from Figure 3.
- Retrieval results in `index.html` are transcribed from Tables 1 and 2.
- The static proving summary table in `index.html` combines Tables 3–5,
  preserving each evaluation set, memory setting, reflection budget, and relative
  cost. Benchmark sizes and aggregate solved totals are omitted from the page. Bold denotes
  the best solved counts within each memory setting, including ties. Cost
  values have a uniform weight, without a special best-value highlight. The
  wide table fits desktop screens; a single vertically grouped table shows the
  same results on narrow screens without horizontal scrolling. All proving
  results remain readable without JavaScript.
- The page lists the public authors in the order supplied by the project owner,
  with UCLA PlusLab as the shared affiliation. Names have no links or affiliation markers.
  The bundled PDF remains the unchanged anonymous submission supplied initially.
- The visual layout is inspired by <https://jianzhubao.github.io/SPACE/>.
  Its scientific content, branding, and author information are not reused.

## Deployment

In repository **Settings → Pages → Build and deployment**, set **Source** to
**GitHub Actions**. This one-time repository setting must be enabled by an
administrator; SSH push access alone cannot change it.

`.github/workflows/website.yml` publishes **only `docs/website`** at the project
URL. It runs on website changes pushed to `main`, and can also be run manually
from the Actions tab. It does not publish the repository root. The uploaded
artifact deliberately flattens the source path, so the public URL is
`/LeanGraphSearch/`, not `/LeanGraphSearch/docs/website/`.

For future edits, change these static files and push. If content changes, verify
the numbers against the cited paper and inspect desktop and mobile layouts.
