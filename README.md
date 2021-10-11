<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">QFUNK</h3>

  <p align="center">
    Quantum information methods that everybody needs. 
    <br />
    <a href="https://github.com/simonsupercool/qfunk"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/simonsupercool/qfunk">View Demo</a>
    ·
    <a href="https://github.com/simonsupercool/qfunk/issues">Report Bug</a>
    ·
    <a href="https://github.com/simonsupercool/qfunk/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <!--<li><a href="#acknowledgements">Acknowledgements</a></li>-->
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This package is intended as a central repository for python code pertaining to quantum information science, primarily the subfield of open quantum system dynamics. 



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

As of the current version, qfunk has only basic dependencies on numpy and scipy. The most recent of these can be downloaded like so
  ```sh
  conda install numpy, scipy
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/simonsupercool/qfunk.git
   ```
2. CD into directory and install package via local pip command
   ```sh
   pip install qfunk
   ```
2. Run the test suite
   ```sh
   python test.py
   ```

If all tests pass then the installation was a success, otherwise please raise an issue that includes the failed test output. 

<!-- USAGE EXAMPLES -->
## Usage

As a general package for quantum information, most standard linear algebra operations are available in qfunk that pertain to this field. On top of this more specific functionality is available, primarily dealing with quantum optics and open quantum systems. Specific examples of which may be found in the examples folder.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- CONTRIBUTING -->
## Contributing

This package is intended as a broad repository for quantum information related code. As such any contribution that meets this criteria are welcome. If you wish to contribute:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/HowCouldYouHaveNotDoneThisAlready`)
3. Commit your Changes (`git commit -m 'Add some HowCouldYouHaveNotDoneThisAlready'`)
4. Push to the Branch (`git push origin feature/HowCouldYouHaveNotDoneThisAlready`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- ACKNOWLEDGEMENTS 
## Acknowledgements

* []()
* []()
* []()


-->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/simonsupercool/qfunk.svg?style=for-the-badge
[contributors-url]: https://github.com/simonsupercool/qfunk/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/simonsupercool/qfunk.svg?style=for-the-badge
[forks-url]: https://github.com/simonsupercool/qfunk/network/members
[stars-shield]: https://img.shields.io/github/stars/simonsupercool/qfunk.svg?style=for-the-badge
[stars-url]: https://github.com/simonsupercool/qfunk/stargazers
[issues-shield]: https://img.shields.io/github/issues/simonsupercool/qfunk.svg?style=for-the-badge
[issues-url]: https://github.com/simonsupercool/qfunk/issues
[license-shield]: https://img.shields.io/github/license/simonsupercool/qfunk.svg?style=for-the-badge
[license-url]: https://github.com/simonsupercool/qfunk/LICENSE.txt
[product-screenshot]: images/screenshot.png