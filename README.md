# my-astro
Former one-off tools made for papers, to be adapted and reused.

## Structure

### Tools sorted per data or object type

For every type of data, I have some tools that find myself constant rewriting or
re-inventing. Examples of those data-type-specific modules are
- `image`: anything generally applicable to astronomical images (some things for IFU images too)
- `spectrum_general`: simple spectra, i.e. anything that is wavelength + flux array.
- `cube`: any tasks involving spectral data cubes that cannot be handled as the
  "spectru_general" case.

Many functions are simple wrappers or shortcuts for existing tools, doing
everything with the formats and settings that I prefer. And having those tools
as a dependency, and these wrapper functions as an example, is useful in itself.

### Other tools

The other module names may not be as obvious. Here is a quick legend.

- `plot`: plotting tools both simple (e.g. my preferred spectrum plotting style)
  and advanced (e.g. display delta arcseconds instead of pixels on the axes of
  an image, make a bivariate color map), but they should be generally
  applicable. For science-specific plots, the other modules may call have
  additional plotting tools.
- `filters`: retrieve filter curves from `stsynphot`, perform synthetic
  photometry, plot a spectrum + filter curves + resulting SED
- `load`: load various file types into my preferred formats. Separate
  functions for e.g. IUE spectra, STIS spectra. Mostly wrappers around astropy
  or specutils io stuff.
- `measure_lines`: The line-measuring methods originally used for the
  PDRs4All MIRI lines paper. This is separate from `spectrum_general`, as the
  latter is mostly for processing spectra. Measuring lines is a task that is
  specific enough to have its own module.
- `mock_sed_fitting`: Simulate fitting of an SED model to noisy SED data.
  Useful for estimating the S/N requirement when planning spectrophotometry
  proposals. Is to be used in combination with the filter objects defined in
  `filters`,
- `regionhacks`: Create regions and use them for a variety of tasks. Also,
  has some utilities to reading them from DS9 region files, and plot them. Most
  of the code leverages the `regions` package as much as possible.
- `rgb`: My script to make RGB images. This was mostly done as a fun
  experiment, and there are likely better scripts. But mine should have some
  reasonable defaults and some useful parameters that can be adjusted from the
  command line.
- `spectral_segments`: Tools to sort, stitch (apply additive/multiplicative
  offsets) to list of spectral segments, and merge them. Mostly used to deal
  with MIRI MRS data, and there is also an N-dimensional merge function which
  should work for data cubes.
- `visualize_footprints`: script to show outlines of the footprints of multiple
  FITS images, on top of a background FITS image.
- `wcshacks`: Read and edit WCS objects. Highlights: rotate an image and create
  new WCS that matches the result, compute field of view size.

## Dev tool notes

- Recently migrated from poetry to uv
- Added some dev dependencies: ruff, ty (new language server to try out)

Check the code and automatic fixes: `ruff check --fix`

Format the code: `black myastro/`

Run scripts: e.g. `uv run python3 myastro/scripts/jwst_fits_info.py`
