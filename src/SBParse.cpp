// Routines to parse a string into an SBProfile

#include <sstream>
#include <fstream>
#include <list>
#include <vector>

// For detailed debugging info:
//#define DEBUGLOGGING

#include "SBParse.h"
#include "StringStuff.h"

#ifdef DEBUGLOGGING
std::ostream* dbgout = &std::cerr;
int verbose_level = 1;
#endif

namespace galsim {

    // Characters that separate words:
    const std::string whitespace=" \t\n\v\f\r";
    // Note also the ( and ) can split words.

    // Return true if the string can be read as number
    // of the type of 2nd argument - and do so.
    template <typename T>
    bool isNumber(const std::string s, T& value) 
    {
        std::istringstream iss(s);
        iss >> value;
        return !iss.fail();
    }

    // Base class for atomic elements of parsed strings
    class Word 
    {
    public:
        virtual std::string print() const=0;
        virtual ~Word() {};
        virtual Word* duplicate() const=0;
    };

    // A word that is still just a string, not yet any special meaning
    class StringWord : public Word, public std::string 
    {
    public:
        StringWord(const StringWord& rhs): std::string(rhs) {}
        StringWord(const std::string& rhs): std::string(rhs) {}
        std::string print() const { return *this; }
        Word* duplicate() const { return new StringWord(*this); }
        ~StringWord() {}
    };

    // An ordered list of words
    class Phrase : public std::list<Word*> 
    {
    public:
        std::string print() const 
        {
            std::string out;
            const_iterator i=begin();
            if (i!=end()) out += (*i)->print();
            for (++i; i!=end(); ++i) out += " " + (*i)->print();
            return out;
        }

        Phrase() {}

        Phrase(const Phrase& rhs) 
        { for (const_iterator i=rhs.begin(); i!=rhs.end(); ++i) push_back((*i)->duplicate()); }

        void clear() 
        {
            for (iterator i=begin(); i!=end(); ++i) delete *i;
            std::list<Word*>::clear();
        }

        ~Phrase() 
        { for (iterator i=begin(); i!=end(); ++i) delete *i; }
    };

    // Key words specified by single character, e.g. our modifiers
    template <char C>
    class KeyWord: public Word 
    {
    public:
        std::string print() const {return std::string(1,C);}
        static bool test(const Word* w) 
        {
            if (dynamic_cast<const KeyWord<C>*> (w)) return true;
            const StringWord* s=dynamic_cast<const StringWord*> (w);
            return (s && nocaseEqual(*s,std::string(1,C)));
        }

        Word* duplicate() const { return new KeyWord; }
        ~KeyWord() {}
    };

    typedef KeyWord<'('> OpenParen;
    typedef KeyWord<')'> CloseParen;
    typedef KeyWord<'*'> ConvolveOp;
    typedef KeyWord<'+'> AddOp;
    typedef KeyWord<'S'> ShearOp;
    typedef KeyWord<'D'> DilateOp;
    typedef KeyWord<'T'> TranslateOp;
    typedef KeyWord<'F'> FluxOp;
    typedef KeyWord<'R'> RotateOp;

    // Break a string into words, breaking at white space and parentheses
    Phrase wordify(const std::string in) 
    {
        Phrase out;
        std::string word;
        for (std::string::const_iterator ip=in.begin(); 
             ip!=in.end();
             ++ip) {
            if (whitespace.find(*ip)!=std::string::npos) {
                // whitespace ends any current word:
                if (!word.empty()) {
                    out.push_back(new StringWord(word));
                    word.clear();
                }
            } else if (*ip=='(') {
                // Special character ends previous word and becomes
                // its own word:
                if (!word.empty()) {
                    out.push_back(new StringWord(word));
                    word.clear();
                }
                out.push_back(new OpenParen);
            } else if (*ip==')') {
                if (!word.empty()) {
                    out.push_back(new StringWord(word));
                    word.clear();
                }
                out.push_back(new CloseParen);
            } else {
                // Regular characters get added to words
                word.push_back(*ip);
            }
        }
        // Save last word if any:
        if (!word.empty()) out.push_back(new StringWord(word));

        return out;
    }

    // Divide a phrase into smaller phrases, divide at locations of work K
    // The words of class K are removed.
    // Breaks are inhibited inside any parentheses.
    // If no works of class K are found, then output list has one element, which
    // is duplicate of the input phrase.
    template <class K>
    std::list<Phrase> SplitAt(const Phrase in) 
    {
        Phrase accum;
        std::list<Phrase> ls;
        int paren_level=0;
        Phrase::const_iterator ip=in.begin();
        while (ip!=in.end()) {
            if (OpenParen::test(*ip)) {
                paren_level++;
                accum.push_back(new OpenParen);
            } else if (CloseParen::test(*ip)) {
                if (paren_level==0) 
                    throw SBError("SBParse unmatched close parenthesis:" + in.print());
                paren_level--;
                accum.push_back(new CloseParen);
            } else if (paren_level==0 && K::test(*ip)) {
                // Divide at un-parenthesized operator:
                ls.push_back(accum);
                accum.clear();
            } else {
                accum.push_back((*ip)->duplicate());
            }
            ++ip;
        }
        if (paren_level!=0) 
            throw SBError("SBParse unmatched open parenthesis:" + in.print());
        ls.push_back(accum);
        return ls;
    }

    // Turn a phrase into an SBProfile, using recursive parsing.
    SBProfile SBParse(Phrase in) 
    {
        if (in.empty())
            throw SBError("SBParse: null expression");

        dbg << "**Starting parser on phrase " << in.print() << std::endl;
        // Strip any bounding parentheses
        while (OpenParen::test(in.front()) && CloseParen::test(in.back())) {
            // Make sure they are matched:  find the CloseParen that matches the initial open
            int paren_level=0;
            Phrase::const_iterator ip=in.begin();
            while (ip!=in.end()) {
                if (OpenParen::test(*ip)) {
                    paren_level++;
                } else if (CloseParen::test(*ip)) {
                if (paren_level==0) 
                    throw SBError("SBParse unmatched close parenthesis:" + in.print());
                paren_level--;
                // If paren_level is back to zero, ip now points to CloseParen matching initial open
                if (paren_level==0) break;
                } 
                ++ip;
            }
            // First make sure that we found a closing match for opener:
            if (ip==in.end())
                throw SBError("SBParse unmatched initial open parenthesis:" + in.print());
            // Now if ip points to last Word, we strip the open and close parentheses:
            ++ip;
            if (ip==in.end()) {
                delete in.front();
                delete in.back();
                in.pop_front();
                in.pop_back();
                if (in.empty())
                    throw SBError("SBParse: empty parentheses");
            } else {
                // If first and last parentheses do not match, stop stripping them:
                break;
            }
        }

        // See if this is a sum expression:
        std::list<Phrase> ls = SplitAt<AddOp>(in);
        if (ls.empty())
            throw SBError("SBParse: no arguments for +");
        if (ls.size()>1) {
            // If so, the output is an SBAdd and call parser recursively on summand Phrases:
            std::list<SBProfile> sum;
#ifdef DEBUGLOGGING
            dbg << "Creating SBAdd from phrases: " << std::endl;
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) 
                dbg << "* " << i->print() << std::endl;
#endif
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) {
                SBProfile summand = SBParse(*i);
                sum.push_back(summand);
            }
            dbg << "**Leaving parser after SBAdd" << std::endl;
            return SBAdd(sum);
        }

        ls.clear();
        // See if this is a convolution expression:
        ls = SplitAt<ConvolveOp>(in);
        if (ls.empty())
            throw SBError("SBParse: no arguments for *");
        if (ls.size()>1) {
            // If so, the output is an SBConvolve and call parser recursively on convolvee Phrases:
            std::list<SBProfile> conv;
#ifdef DEBUGLOGGING
            dbg << "Creating SBConvolve from phrases: " << std::endl;
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) 
                dbg << "* " << i->print() << std::endl;
#endif
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) {
                SBProfile term = SBParse(*i);
                conv.push_back(term);
            }
            dbg << "**Leaving parser after SBConvolve" << std::endl;
            return SBConvolve(conv);
        }

        // else: any modifiers (from RHS)?
        Phrase args;
        while (!in.empty()) {
            Word* i = in.back();
            in.pop_back();
            if (ShearOp::test(i) || DilateOp::test(i) || 
                TranslateOp::test(i) || FluxOp::test(i) || RotateOp::test(i)) {

                // Found a modifier.  Parse the LHS and apply modification
                dbg << "Found modifier, phrase to modify is " << in.print() << std::endl;
                SBProfile base = SBParse(in);
                // Apply appropriate modification:
                Phrase::iterator ia=args.begin();
                if (ShearOp::test(i)) {
                    delete i;
                    double g1, g2;
                    if (args.size()!=2 
                        || !isNumber((*(ia++))->print(),g1)
                        || !isNumber((*ia)->print(),g2))
                        throw SBError("SBParse: bad arguments for shear: " + args.print());
                    dbg << "** Leaving SBParse after shearing by " << g1 << " " << g2 << std::endl;
                    base.applyShear(g1, g2);
                    return base;
                } else if (DilateOp::test(i)) {
                    delete i;
                    double f;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),f))
                        throw SBError("SBParse: bad arguments for dilation: " + args.print());
                    Ellipse e(Shear(), std::log(f), Position<double>());
                    dbg << "** Leaving SBParse after dilating by " << f << std::endl;
                    base.applyTransformation(e);
                    return base;
                } else if (TranslateOp::test(i)) {
                    delete i;
                    double dx,dy;
                    if (args.size()!=2 
                        || !isNumber((*(ia++))->print(),dx) 
                        || !isNumber((*ia)->print(),dy))
                        throw SBError("SBParse: bad arguments for translation: " + args.print());
                    dbg << "** Leaving SBParse after translating by " << dx << " " 
                              << dy << std::endl;
                    base.applyShift(dx,dy);
                    return base;
                } else if (RotateOp::test(i)) {
                    // TODO: Not sure how much we're planning on using SBParse,
                    // but if we are, it would be nice to have theta specified with units.
                    delete i;
                    double theta;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),theta))
                        throw SBError("SBParse: bad arguments for rotation: " + args.print());
                    dbg << "** Leaving SBParse after rotating by " << theta << std::endl;
                    base.applyRotation(theta * radians);
                    return base;
                } else if (FluxOp::test(i)) {
                    delete i;
                    double f;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),f))
                        throw SBError("SBParse: bad arguments for flux: " + args.print());
                    dbg << "** Leaving SBParse after flux set to " << f << std::endl;
                    base.setFlux(f);
                    return base;
                }
            } else {
                // This Word is not a modifier.  Add to argument list
                args.push_front(i);
            }
        }

        // else: should be a primitive, specified by first word and rest are arguments. Build and return
        int nargs = args.size()-1;
        // Translate arguments into doubles since that's what most primitives want.
        bool allNumbers=true;
        std::vector<double> dargs;
        double vv;
        for (Phrase::const_iterator ip=++args.begin(); ip!=args.end(); ++ip) {
            allNumbers &= isNumber((*ip)->print(), vv);
            dargs.push_back(vv);
        }

        std::string sbtype=args.front()->print();
        if (nocaseEqual(sbtype, "gauss")) {
            // Gaussian: args are [sigma=1]
            if (nargs>1 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBGaussian: " + args.print());
            double flux=1.;
            double sigma = (nargs>0) ? dargs[0] : 1.;
            dbg << "**Returning gaussian with flux, sigma " << flux << " " << sigma << std::endl;
            return SBGaussian(flux, sigma);

        } else if (nocaseEqual(sbtype, "exp")) {
            // Exponential Disk: args [re=1.]
            if (nargs>1 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBExponential: " + args.print());
            double flux=1.;
            double re = (nargs>0) ? dargs[0] : 1.;
            dbg << "**Returning exp with flux, re " << flux << " " << re << std::endl;
            return SBExponential(flux, re/1.67839);

        } else if (nocaseEqual(sbtype, "sersic")) {
            // Sersic: args are [n] [re=1]
            if (nargs<1 || nargs>2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBSersic: " + args.print());
            double flux=1.;
            double n=dargs[0];
            double re = (nargs>1) ? dargs[1] : 1.;
            dbg << "**Returning sersic with n, flux, re " << n 
                << " " << flux << " " << re << std::endl;
            return SBSersic(n, flux, re);

        } else if (nocaseEqual(sbtype, "box")) {
            // Sersic: args are [xw=1] [yw=xw]
            if (nargs>2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBBox: " + args.print());
            double flux=1.;
            double xw =(nargs>0) ? dargs[0] : 1.;
            double yw = (nargs>1) ? dargs[1] : xw;
            dbg << "**Returning box with xw, yw, flux " << xw
                << " " << yw << " " << flux << std::endl;
            return SBBox(xw,yw,flux);

        } else if (nocaseEqual(sbtype, "airy")) {
            // Airy: args are [D/lambda] [obscuration]
            if (nargs!=2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBAiry: " + args.print());
            double flux=1.;
            double D = dargs[0];
            double obs = dargs[1];
            dbg << "**Returning airy with D, obs, flux " << D
                << " " << obs << " " << flux << std::endl;
            return SBAiry(D,obs,flux);

        } else if (nocaseEqual(sbtype, "moffat")) {
            // Airy: args are [beta] [truncationFWHM] [re=1]
            if (nargs<2 || nargs>3 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBMoffat: " + args.print());
            double flux=1.;
            double beta = dargs[0];
            double truncationFWHM = dargs[1];
            double re = (nargs>2) ? dargs[2] : 1.;
            dbg << "**Returning moffat with beta, truncation, flux, re " << beta
                << " " << truncationFWHM << " " << flux << " " << re << std::endl;
            return SBMoffat(beta, truncationFWHM, flux, re);

        } else if (nocaseEqual(sbtype, "laguerre")) {
            // Laguerre: args are [filename]
            if (nargs!=1)
                throw SBError("SBParse: missing filename for SBLaguerre: " + args.print());
            std::string psfName = (*(++args.begin()))->print(); 
            LVector bPSF;
            std::ifstream cpsf(psfName.c_str());
            if (!cpsf)
                throw SBError("SBParse could not open Laguerre PSF file " + psfName);
            std::string buffer;
            getlineNoComment(cpsf, buffer);
            std::istringstream iss(buffer);
            double g1, g2, mu;
            if (!(iss >> g1 >> g2 >> mu))
                throw SBError("SBParse error on Laguerre basis ellipse: " + buffer);
            if (!(cpsf >> bPSF))
                throw SBError("SBParse error reading Laguerre PSF file " + psfName);

            SBLaguerre sbl(bPSF, std::exp(mu));
            sbl.applyShear(g1, g2);
            return sbl;
        } else {
            throw SBError("SBParse: unrecognized primitive type: " + sbtype);
        }
    }

    SBProfile SBParse(std::string instring) 
    {
        Phrase in = wordify(instring);
        return SBParse(in);
    }

}

