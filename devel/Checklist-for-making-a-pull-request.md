If you've completed some piece of code that you want to have merged into master, below is a checklist for what you should do on the branch containing the new code before making a pull request (or after making it, if you do any substantial overhaul to the code then you should redo these steps):

## Getting ready for a pull request:

* Get the branch up to date, by merging in any recent changes from the master branch.
* A final check that you can build the code: "scons", "scons install"
* Make sure the unit tests still work: "scons tests".  Ideally there should be some new unit tests for the new functionality, unless the work is completely covered by existing unit tests.
* You don't necessarily need to test on Python 2.6 or Python 3.4, but things to watch out for w.r.t. compatibility with these are:
   * (2.6) Use numbers in the braces for string format statements.  e.g. `"{0}".format(x)`, not `"{}".format(x)`
   * (3.4) Use print function, for all print statements.  e.g. `print('x = ',x)`, not `print 'x = ',x`.  Add `from __future__ import print_function` at the top of the file if it's not there yet.
   * (3.4) Use `//` for any division where you want the result to be an integer.  e.g. `imid = (i1 + i2)//2`, not `imid = (i1 + i2)/2`.  The latter will be a float in Python 3.
   * (3.4) Use simpler names of things like `range`, `map`, `zip`, `values`, `items` rather than the longer names `xrange`, `imap`, `izip`, `itervalues`, `iteritems`.  If you think the performance impact is important, see [this guide](http://python-future.org/compatible_idioms.html) for the Python 2/3 compatible syntax.
   * (3.4) Similarly, use `str`, not `basestring`.  And if you really need it to be `basestring`, see [this guide](http://python-future.org/compatible_idioms.html) for the right syntax to use.
* Check the documentation for your code at the C++ and python level to make sure it's complete and up-to-date.
* Update CHANGELOG.md to mention your change.
* Make sure that the new code conforms to our code guidelines, for example no tabs (4 spaces), 100 character line width.
* If the change is sufficiently major, update the quick reference guide in doc/ref/ (the .tex file) and the PDF version in doc/ .
* Make sure you can still run all demos using python and config.  The easiest way is: "cd examples", "./check_yaml", "./check_json".
* If you updated the tutorial functionality, update the [tutorial wiki page](https://github.com/GalSim-developers/GalSim/wiki/Tutorials).
* Make sure any new files have BSD license at the top.
* Depending on the nature of the change, check whether updates to README and INSTALL are needed.

## Making the request

* On the GitHub site, go to "Code".  Then click the green "Compare and Review" button.  Your branch is probably in the "Example Comparisons" list, so click on it.  If not, select it for the "compare" branch.
* Make sure you are comparing your new branch to master.  It probably won't be, since the front page is the latest release branch, rather than master now.  So click the base branch and change it to master.
* Press Create Pull Request button.
* Give a brief title. (We usually leave the branch number as the start of the title.)
* Explain the major changes you are asking to be code reviewed.  Often it is useful to open a second tab in your browser where you can look through the diff yourself to remind yourself of all the changes you have made.

## After submitting the pull request

* Check to make sure that the PR can be merged cleanly.  If it can, GitHub will report that "This branch has no conflicts with the base branch."  If it doesn't, then you need to merge from master into your branch and resolve any conflicts.
* Wait 20 minutes or so for the continuous integration tests to be run.  Then make sure that the Travis-CI test reports no errors.  If not, click through to the details and try to figure out what is causing the error and fix it.
* Check that CodeCov patch test reports "100% of diff hit".  If it doesn't, then click through to the details and look for pink lines, which are lines in the code that you changed, but for which you don't have unit tests.  Ideally, add unit tests to cover all lines, and all branches of if statements. (I.e. make sure it evaluates to both true and false at some point -- if it doesn't, CodeCov will mark it yellow.)  If you aren't sure what to do to test something, it's fine to make a comment on the PR page asking for advice.

## After code review:

* Once at least 1 and preferably 2 people have reviewed the code, and you have responded to all of their comments, we generally solicit for "any other comments" and give people a few more days before merging.
* Click the "Merge pull request" button at the bottom of the PR page.
* Click the "Delete branch" button.
* Check the results of the Travis CI tests: https://travis-ci.org/GalSim-developers/GalSim/builds
* If there are other open pull requests, it is generally good etiquette to merge your changes into those branches.
* Please do not merge the new changes into other open branches that are not yet in PR.  (Unless you are the principal developer on the branch of course.)  Developers working on those branches may have unpushed commits, so you should let them do the merge from master when they are ready for it.
