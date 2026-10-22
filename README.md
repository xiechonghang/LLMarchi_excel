# Local main branch history backup

This private backup preserves the 34 original project commits through `ffb85e52451de36341ccfc811242378b4b80f515`. Only `refs/heads/main` is included; internal tool refs are excluded.

The source snapshot on the remote `main` branch has the exact same Git tree as that local commit: `da3cee9983ef282e23fbee440c25ba239dabb53c`. The remote snapshot commit differs because it was uploaded through the authorized GitHub connector when local Git authentication was unavailable.

To restore the original history, download the three parts together, then run:

```sh
cat LLMarchi_excel_main_history.bundle.part01 LLMarchi_excel_main_history.bundle.part02 LLMarchi_excel_main_history.bundle.part03 > LLMarchi_excel_main_history.bundle
shasum -a 256 LLMarchi_excel_main_history.bundle
git clone -b main LLMarchi_excel_main_history.bundle LLMarchi_excel-restored
```

Expected SHA-256: `04fc08b4c5cd1be2207f2268db499c1547a5a14db046cdedb3db940d85e2ea4e`.

Models, generated outputs, environments, ignored coordination evidence, and the two uncommitted local document deletions are outside this backup, following the existing project scope.
