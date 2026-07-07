const { getDefaultConfig } = require("expo/metro-config");
const path = require("path");

const config = getDefaultConfig(__dirname);

// Monorepo: let Metro see the workspace packages and the native module.
config.watchFolders = [path.resolve(__dirname, "../..")];

module.exports = config;
