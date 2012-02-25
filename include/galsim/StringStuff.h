
// Common convenience functions with strings

#ifndef STRINGSTUFF_H
#define STRINGSTUFF_H

#include <iostream>
#include <sstream>
#include <string>

namespace sbp {

    extern bool isComment(const std::string& instr);

    extern std::istream& getlineNoComment(std::istream& is, std::string& s);

    bool nocaseEqual(char c1, char c2);
    bool nocaseEqual(const std::string& s1, const std::string& s2);

    // Remove trailing whitespace
    void stripTrailingBlanks(std::string& s);
    // Remove everything after & including the last "." period
    void stripExtension(std::string& s);

    // Make a string that holds the current time and the command line
    std::string taggedCommandLine(int argc, char *argv[]);
}

#endif
