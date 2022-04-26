Version 1.0.2
-------------

* Fix missing conjugation in complex-valued inverse transform
* Fix a wrong assertion in debug builds using the fixed-size version
* Output exception messages to stderr when using the C interface
* Parallelize more pre- and postprocessing loops
* Add optional benchmark programs

Version 1.0.1
-------------

* Add CMake configuration files, library can now be imported using `find_package(SFT REQUIRED)`
* Add interface targets sft_fixed and sftf_fixed for use with `target_link_libraries()`

Version 1.0
-----------

* First public release
