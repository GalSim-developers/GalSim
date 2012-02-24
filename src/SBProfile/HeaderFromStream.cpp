#include "HeaderFromStream.h"
#include "Std.h"

namespace galsim {

    HdrRecordBase* ReadASCIIHeader(std::string in) 
    {
        const char quote='\''; //will bound a quoted string keyword or value
        const std::string comments="/"; //starts comment (unless quoted)
        const char eq='='; //can be present after keyword

        std::string keyword;
        std::string vstring;
        std::string comment;

        std::string::size_type i=0;
        std::string::size_type l=in.length();

        // skip opening white space:
        while (i<l && isspace(in[i])) ++i;
        if (i==l) return 0;

        //Keyword ends at next white, =, comment, or end:
        while (i<l 
               && in[i]!=quote
               && in[i]!=eq
               && !isspace(in[i])
               && comments.find(in[i])==std::string::npos ) keyword+=in[i++];
        if (keyword.length()==0) return 0; //no useful info.

        //std::cerr << "Working on keyword <" << keyword << ">" << std::endl;
        // Skip whitespace or equals; done for end or comment
        while (i<l && (in[i]==eq || isspace(in[i])) ) ++i;

        // Null record if we have nothing or comment left:
        if (i==l) return new HdrRecordNull(keyword);
        if (comments.find(in[i])!=std::string::npos) 
            return new HdrRecordNull(keyword, in.substr(i+1));

        // A keyword named "HISTORY" or "COMMENT" is all comment, really
        if (keyword=="COMMENT" || keyword=="HISTORY")
            return new HdrRecordNull(keyword, in.substr(i));

        // A quoted value is string:
        if (in[i]==quote) {
            //If value is quoted, get everything until next quote
            ++i; //skip the quote
            while (i<l && in[i]!=quote) vstring+=in[i++];
            if (i==l) return 0; // unbounded quote, failure!!!
            ++i; //skip the closing quote
            while (i<l && isspace(in[i])) i++; // skip whitespace
            //std::cerr << "Got string: " << vstring << std::endl;
            if (i==l) return new HdrRecord<std::string>(keyword, vstring);
            else if (comments.find(in[i])!=std::string::npos) // Comment left?
                return new HdrRecord<std::string>(keyword, vstring, in.substr(i+1));
            else return 0; // ??? failure - something other than comment after string
        }

        if (in[i]=='T' || in[i]=='F') {
            // Boolean valued:
            bool value= (in[i]=='T');
            i++;
            while (i<l && isspace(in[i])) i++; // skip whitespace
            if (i==l) return new HdrRecord<bool>(keyword, value);
            else if (comments.find(in[i])!=std::string::npos) // Comment left?
                return new HdrRecord<bool>(keyword, value, in.substr(i+1));
            else return 0; // ??? failure - something other than comment after T/F
        }

        // Otherwise we are getting either an integer or a float (ignore complex)
        while (i<l && comments.find(in[i])==std::string::npos ) vstring+=in[i++];
        // Strip trailing whitespace if was not a quoted response:
        std::string::size_type pos = vstring.size();
        while (pos > 0 && isspace(vstring[pos - 1])) pos--;
        vstring.erase(pos);
        //std::cerr << "Getting value from string: " << vstring << std::endl;
        // Collect comment
        if (comments.find(in[i])!=std::string::npos) // Comment left?
            comment = in.substr(i+1);
        HdrRecord<int>* hi = new HdrRecord<int>(keyword, 0, comment);
        if (!hi->setValueString(vstring)) return hi;
        // If that failed, try a double
        HdrRecord<double>* hd = new HdrRecord<double>(keyword, 0., comment);
        if (!hd->setValueString(vstring)) return hd;
        else return 0; // Formatting error
    }

    ImageHeader HeaderFromStream(std::istream& is) 
    {
        ImageHeader h;
        std::string buffer;
        while (getline(is, buffer)) {
            HdrRecordBase* hrb = ReadASCIIHeader(buffer);
            if (!hrb) 
                FormatAndThrow<ImageError>() << "Bad ASCII header card <" << buffer << ">";
            if (hrb->getKeyword()=="END") return h;
            else if (hrb->getKeyword()=="COMMENT") h.addComment(hrb->getComment());
            else if (hrb->getKeyword()=="HISTORY") h.addHistory(hrb->getComment());
            else h.append(hrb);
        }
        return h;
    }

    std::ostream& operator<<(std::ostream& os, const ImageHeader& h) 
    {
        for (h.rewind(); !h.atEnd(); h.incr())
            os << h.current()->writeCard() << std::endl;
        os << "END     " << std::endl;
        return os;
    }

}
