// Routines for parameter sets
#include <string>
//#include <cctype>
#include "Pset.h"

namespace sbp {

    void Pset::addMemberNoValue(const char *k, const int _f, const char *c) 
    { l.push_back( new PsetMember(k, _f, c) ); }


    //??? discern null strings from default string request?

    // Take an input line, pull out strings representing keyword & value.
    // Returns true if this line has information.  value string may be empty
    // if no value is specified, but keyword should always have something.
    // Throw exception for an unterminated quoted string.
    bool Pset::read_keyvalue(const std::string& in, std::string& keyword, std::string& value) 
    {
        const char quote='"'; //will bound a quoted string keyword or value
        const std::string comments="#;"; //starts comment (unless quoted)
        const char eq='='; //can be present after keyword

        keyword.resize(0);
        value.resize(0);

        std::string::size_type i=0;
        std::string::size_type l=in.length();

        // skip opening white space:
        while (i<l && isspace(in[i])) ++i;
        if (i==l) return false;


        if (in[i]==quote) {
            //If keyword is quoted, get everything until next quote
            ++i; //skip the quote
            while (i<l && in[i]!=quote) keyword+=in[i++];
            if (i==l) throw PsetUnboundedQuote(in);
            ++i; //skip the quote
        } else {
            //Keyword ends at next white, =, comment, or end:
            while (i<l 
                   && in[i]!=quote
                   && in[i]!=eq
                   && !isspace(in[i])
                   && comments.find(in[i])==std::string::npos ) keyword+=in[i++];
        }
        if (keyword.length()==0) return false; //no useful info.

        // Skip whitespace or equals; done for end or comment
        while (i<l && (in[i]==eq || isspace(in[i])) ) ++i;

        if (i==l) return true; //end - keyword with no value.
        if (comments.find(in[i])!=std::string::npos) return true; //comment

        // Get the value string:
        if (in[i]==quote) {
            //If value is quoted, get everything until next quote
            ++i; //skip the quote
            while (i<l && in[i]!=quote) value+=in[i++];
            if (i==l) throw PsetUnboundedQuote(in);
            ++i; //skip the closing quote
        } else {
            //Value ends at comment, or end:
            while (i<l 
                   && comments.find(in[i])==std::string::npos ) value+=in[i++];
            // Strip trailing whitespace if was not a quoted response:
            std::string::size_type pos = value.size();
            while (pos > 0 && isspace(value[pos - 1])) pos--;
            value.erase(pos);
        }

        return true; //could warn here if there are extra characters...
    }

}
