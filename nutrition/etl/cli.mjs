import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { buildBundle } from "./build-bundle.mjs";

const { values } = parseArgs({
  options: {
    "fdc-dir": { type: "string" },
    out: { type: "string", default: "nutrient-bundle.sqlite" },
    // priors.json from model/priors/fit_priors.py — seeds the shape_priors table.
    priors: { type: "string" },
  },
});

if (!values["fdc-dir"]) {
  console.error(
    "Usage: node etl/cli.mjs --fdc-dir <dir with FDC csv files> [--out bundle.sqlite] [--priors priors.json]",
  );
  console.error(
    "Download the CSVs (Foundation, SR Legacy, FNDDS) from https://fdc.nal.usda.gov/download-datasets/",
  );
  process.exit(1);
}

const priors = values.priors ? JSON.parse(readFileSync(values.priors, "utf8")) : null;
const stats = buildBundle({ fdcDir: values["fdc-dir"], out: values.out, priors });
console.log(
  `bundle written to ${values.out}: ${stats.foods} foods, ` +
    `${stats.withDensity} with portion-derived density, ` +
    `${stats.shapePriors} shape priors, fts=${stats.fts}`,
);
