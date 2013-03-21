# Default the install to dummy locations
debug_install_dir = 'install_debug'
release_install_dir = 'install_release'

# Only use the correct install path for the true variant
if 'install-debug' in COMMAND_LINE_TARGETS:
    debug_install_dir = Dir('#/rftk/native').abspath
else:
    release_install_dir = Dir('#/rftk/native').abspath

SConscript('modules/SConscript', variant_dir='build/debug', duplicate=True,
    exports={'install_path': debug_install_dir, 'variant': 'debug'})
SConscript('modules/SConscript', variant_dir='build/release', duplicate=True,
    exports={'install_path': release_install_dir, 'variant': 'release'})

Default('install-debug', 'install-release')