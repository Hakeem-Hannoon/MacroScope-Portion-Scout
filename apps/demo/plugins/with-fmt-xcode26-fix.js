// Xcode 26 clang rejects the consteval usage in the fmt 11.x pod that React
// Native vendors (facebook/react-native#55601). Compiling just the fmt pod as
// C++17 sidesteps it (consteval is C++20-only). The ios/ directory is
// generated (CNG), so the Podfile patch has to be applied at prebuild time.
// Remove once RN vendors fmt >= 12.1 (RN 0.83.9+ / Expo SDK 56).
const { withDangerousMod } = require('expo/config-plugins');
const fs = require('fs');
const path = require('path');

const MARKER = "target.name == 'fmt'";
const SNIPPET = `
    # Xcode 26 clang rejects fmt 11's consteval usage; C++17 has no consteval.
    # Remove once RN vendors fmt >= 12.1 (RN 0.83.9+).
    installer.pods_project.targets.each do |target|
      if ${MARKER}
        target.build_configurations.each do |config|
          config.build_settings['CLANG_CXX_LANGUAGE_STANDARD'] = 'c++17'
        end
      end
    end
`;

module.exports = function withFmtXcode26Fix(config) {
  return withDangerousMod(config, [
    'ios',
    (config) => {
      const podfilePath = path.join(config.modRequest.platformProjectRoot, 'Podfile');
      let contents = fs.readFileSync(podfilePath, 'utf8');
      if (!contents.includes(MARKER)) {
        const anchor = /post_install do \|installer\|/;
        if (!anchor.test(contents)) {
          throw new Error('with-fmt-xcode26-fix: no post_install block found in Podfile');
        }
        contents = contents.replace(anchor, (match) => `${match}\n${SNIPPET}`);
        fs.writeFileSync(podfilePath, contents);
      }
      return config;
    },
  ]);
};
