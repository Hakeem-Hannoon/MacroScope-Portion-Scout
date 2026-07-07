import { parseArgs } from "node:util";
import { buildBundle } from "./build-bundle.mjs";

const { values } = parseArgs({
  options: {
    "fdc-dir": { type: "string" },
    out: { type: "string", default: "nutrient-bundle.sqlite" },
  },
});

if (!values["fdc-dir"]) {
  console.error(
    "Usage: node etl/cli.mjs --fdc-dir <dir with FDC csv files> [--out bundle.sqlite]",
  );
  console.error(
    "Download the CSVs (Foundation, SR Legacy, FNDDS) from https://fdc.nal.usda.gov/download-datasets/",
  );
  process.exit(1);
}

const stats = buildBundle({ fdcDir: values["fdc-dir"], out: values.out });
console.log(
  `bundle written to ${values.out}: ${stats.foods} foods, ` +
    `${stats.withDensity} with portion-derived density, fts=${stats.fts}`,
);
