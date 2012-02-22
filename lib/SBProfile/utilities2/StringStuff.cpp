// $Id: StringStuff.cpp,v 1.1.1.1 2009/10/30 21:20:52 garyb Exp $
#include "StringStuff.h"
#include <cctype>
#include <ctime>

namespace stringstuff {
  bool isComment(const string& instr) {
    std::istringstream is(instr);
    string word1;
    is >> word1;
    return (!is || word1.empty() || word1[0]=='#');
  }

  std::istream& getlineNoComment(std::istream& is, string& s) {
    do {
      if (!getline(is,s)) return is;
    } while (isComment(s));
    return is;
  }

  bool nocaseEqual(char c1, char c2) {
    return std::toupper(c1)==std::toupper(c2);
  }

  bool nocaseEqual(const string& s1, const string& s2) {
    if (s1.size() != s2.size()) return false;
    string::const_iterator p1=s1.begin();
    string::const_iterator p2=s2.begin();
    for ( ; p1!=s1.end(); ++p1, ++p2)
      if (!nocaseEqual(*p1, *p2)) return false;
    return true;
  }

  void stripTrailingBlanks(string& s) {
    string::iterator tail;
    while (!s.empty()) {
      tail=s.end()-1;
      if (!std::isspace(*tail)) return;
      s.erase(tail);
    }
  }

  void stripExtension(string& s) {
    size_t dot=s.find_last_of(".");
    if (dot==string::npos) return;	// No extension
    s.erase(dot);
  }

  std::string taggedCommandLine(int argc, char *argv[]) {
    time_t now;
    time(&now);
    string output = ctime(&now);
    // get rid of last character, which is linefeed:
    output.erase(output.size()-1);  // ????
    output += ": ";
    for (int i=0; i<argc; i++) {
      output += " "; output += argv[i];
    }
    return output;
  }
}
