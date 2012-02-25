// Routines to parse a string into an SBProfile

//#include <cctype>
#include <sstream>
#include <fstream>
#include <list>
#include <vector>

#include "SBParse.h"
#include "StringStuff.h"

namespace sbp {

    // Characters that separate words:
    const std::string whitespace=" \t\n\v\f\r";
    // Note also the ( and ) can split words.

    template <typename T>
    bool isNumber(const std::string s, T& value) 
    {
        std::istringstream iss(s);
        iss >> value;
        return !iss.fail();
    }

    class Word 
    {
    public:
        virtual std::string print() const=0;
        virtual ~Word() {};
        virtual Word* duplicate() const=0;
    };

    class StringWord : public Word, public std::string 
    {
    public:
        StringWord(const StringWord& rhs): std::string(rhs) {}
        StringWord(const std::string& rhs): std::string(rhs) {}
        std::string print() const {return *this;}
        Word* duplicate() const {return new StringWord(*this);}
        ~StringWord() {}
    };

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
                    throw SBError("SBParse unmatched parentheses:" + in.print());
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
            throw SBError("SBParse unmatched parentheses:" + in.print());
        ls.push_back(accum);
        return ls;
    }

    SBProfile* SBParse(Phrase in) 
    {
        if (in.empty())
            throw SBError("SBParse: null expression");

        // Strip any bounding parentheses
        while (OpenParen::test(in.front()) && CloseParen::test(in.back())) {
            delete in.front();
            delete in.back();
            in.pop_front();
            in.pop_back();
            if (in.empty())
                throw SBError("SBParse: null expression");
        }

        // See if this is a sum expression:
        std::list<Phrase> ls = SplitAt<AddOp>(in);
        if (ls.empty())
            throw SBError("SBParse: no arguments for +");
        if (ls.size()>1) {
            SBAdd* sba = new SBAdd;
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) {
                SBProfile* summand = SBParse(*i);
                sba->add(*summand);
                delete summand;
            }
            return sba;
        }

        ls.clear();
        // See if this is a convolution expression:
        ls = SplitAt<ConvolveOp>(in);
        if (ls.empty())
            throw SBError("SBParse: no arguments for *");
        if (ls.size()>1) {
            SBConvolve* sbc = new SBConvolve;
            for (std::list<Phrase>::iterator i=ls.begin(); i!=ls.end(); ++i) {
                SBProfile* term = SBParse(*i);
                sbc->add(*term);
                delete term;
            }
            return sbc;
        }

        // else: any modifiers (from RHS)?
        Phrase args;
        while (!in.empty()) {
            Word* i = in.back();
            in.pop_back();
            if (ShearOp::test(i) || DilateOp::test(i) || 
                TranslateOp::test(i) || FluxOp::test(i) || RotateOp::test(i)) {

                // Found a modifier.  Parse the LHS and apply modification
                SBProfile* base = SBParse(in);
                // Apply appropriate modification:
                Phrase::iterator ia=args.begin();
                if (ShearOp::test(i)) {
                    delete i;
                    double e1, e2;
                    if (args.size()!=2 
                        || !isNumber((*(ia++))->print(),e1) 
                        || !isNumber((*ia)->print(),e2))
                        throw SBError("SBParse: bad arguments for shear: " + args.print());
                    SBProfile* out = base->shear(e1,e2);
                    delete base;
                    return out;
                } else if (DilateOp::test(i)) {
                    delete i;
                    double f;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),f))
                        throw SBError("SBParse: bad arguments for dilation: " + args.print());
                    Ellipse e(0., 0., log(f));
                    SBProfile* out = base->distort(e);
                    delete base;
                    return out;
                } else if (TranslateOp::test(i)) {
                    delete i;
                    double dx,dy;
                    if (args.size()!=2 
                        || !isNumber((*(ia++))->print(),dx) 
                        || !isNumber((*ia)->print(),dy))
                        throw SBError("SBParse: bad arguments for translation: " + args.print());
                    SBProfile* out = base->shift(dx,dy);
                    delete base;
                    return out;
                } else if (RotateOp::test(i)) {
                    delete i;
                    double theta;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),theta))
                        throw SBError("SBParse: bad arguments for rotation: " + args.print());
                    SBProfile* out = base->rotate(theta);
                    delete base;
                    return out;
                } else if (FluxOp::test(i)) {
                    delete i;
                    double f;
                    if (args.size()!=1
                        || !isNumber((*ia)->print(),f))
                        throw SBError("SBParse: bad arguments for flux: " + args.print());
                    base->setFlux(f);
                    return base;
                }
            } else {
                // This Word is not a modifier.  Add to argument list
                args.push_front(i);
            }
        }

        // else: should be a primitive. Build and return
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
            return new SBGaussian(flux, sigma);

        } else if (nocaseEqual(sbtype, "exp")) {
            // Exponential Disk: args [re=1.]
            if (nargs>1 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBExponential: " + args.print());
            double flux=1.;
            double re = (nargs>0) ? dargs[0] : 1.;
            return new SBExponential(flux, re/1.67839);

        } else if (nocaseEqual(sbtype, "sersic")) {
            // Sersic: args are [n] [re=1]
            if (nargs<1 || nargs>2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBSersic: " + args.print());
            double flux=1.;
            double n=dargs[0];
            double re = (nargs>1) ? dargs[1] : 1.;
            return new SBSersic(n, flux, re);

        } else if (nocaseEqual(sbtype, "box")) {
            // Sersic: args are [xw=1] [yw=xw]
            if (nargs>2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBBox: " + args.print());
            double flux=1.;
            double xw =(nargs>0) ? dargs[0] : 1.;
            double yw = (nargs>1) ? dargs[1] : xw;
            return new SBBox(xw,yw,flux);

        } else if (nocaseEqual(sbtype, "airy")) {
            // Airy: args are [D/lambda] [obscuration]
            if (nargs!=2 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBBox: " + args.print());
            double flux=1.;
            double D = dargs[0];
            double obs = dargs[1];
            return new SBAiry(D,obs,flux);

        } else if (nocaseEqual(sbtype, "moffat")) {
            // Airy: args are [beta] [truncationFWHM] [re=1]
            if (nargs<2 || nargs>3 || !allNumbers)
                throw SBError("SBParse: Bad arguments for SBMoffat: " + args.print());
            double flux=1.;
            double beta = dargs[0];
            double truncationFWHM = dargs[1];
            double re = (nargs>2) ? dargs[2] : 1.;
            return new SBMoffat(beta, truncationFWHM, flux, re);

#ifdef USE_LAGUERRE
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
            double e1, e2, mu;
            if (!(iss >> e1 >> e2 >> mu))
                throw SBError("SBParse error on Laguerre basis ellipse: " + buffer);
            if (!(cpsf >> bPSF))
                throw SBError("SBParse error reading Laguerre PSF file " + psfName);

            SBLaguerre sbl(bPSF, exp(mu));
            return sbl.shear(e1, e2);
#endif
        } else {
            throw SBError("SBParse: unrecognized primitive type: " + sbtype);
        }
    }

    SBProfile* SBParse(std::string instring) 
    {
        Phrase in = wordify(instring);
        return SBParse(in);
    }

}

