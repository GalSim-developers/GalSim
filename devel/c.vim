" Mike Jarvis's file for formatting LSST C++ code.
" This file should go in ~/.vim/ftplugin/c.vim
" And in your .vimrc file, you should have the following line:
" filetype plugin on
"
" In addition to the formatting specified by the standards document at:
" http://dev.lsstcorp.org/trac/wiki/C%2B%2BStandard
" this file makes the following choices: 
" (with examples from actual LSST code, which motivated the choices)
"
" 1) Indent a template <> line at the same indent as the following
"    function declaration. 
"
" e.g.:
"
" template<typename ReturnT>
" class Function : public lsst::daf::data::LsstBase {
"
" 2) For a multi-line template <> specification, align the 
"    second and later lines with the space after the opening <. e.g.:
"
" template<typename ImageT, typename MaskT=lsst::afw::image::MaskPixel,
"          typename VarianceT=lsst::afw::image::VariancePixel>
"                   
" 3) For a multi-line parenthesis group with a closing ) starting
"    the last line, line up the ) at the same indent as the first 
"    line with the (.  e.g.:
"
" void convolve(
"     OutImageT& convolvedImage,
"     InImageT const& inImage,
"     KernelT const& kernel,
"     bool doNormalize,
"     bool copyEdge = false
" );
"
" 4) For the initialization line in a class constructor definition,
"    we allow two possibilities:
"
"    a) The : and all initializations are on the same line as the function name.
"       e.g.:
"
" Filter() : _id(U) {}
"
"    b) The : ends the line with the function arguments,
"       the initialization parameters are indented 4 spaces,
"       and then the opening brace of the function is on a line by itself.
"       e.g.:
"
" template<typename OtherPixelT>
" ImageBase(const ImageBase<OtherPixelT>& rhs, const bool deep) :
"     lsst::daf::data::LsstBase(typeid(this)) 
" {
"     ...
" }
"
" Note -- This is an exception to the usual rule that the {
" goes at the end of the line, rather than starting a new one.
" This exception is because I cannot find any easy way to get vim
" to format the function definition part without putting an extra
" shiftwidth of indenting.  
"
" As a matter of fact, I would prefer to have all functions
" formatted this way, since I think it is always clearer to have the 
" opening brace for functions on their own line, but that's not the way 
" the LSST standards are currently written.  So here, I am only making 
" an exception for constructors with initialization lists.
"
" 5) For a class whose inheritances require more than one line, it also
"    seems that the opening brace should be on its own line.
"    So we allow for the same two possibilities:
"
"    a) The : and all inheritances are on one line.  e.g.:
"
" template<typename PixelT>
" class Image : public ImageBase<PixelT> {
"     ...
" };
"
"    b) The inheritances are each aligned after the :,
"       and then the opening brace is on the next line. e.g.:
"
" class Exposure : public lsst::daf::base::Persistable,
"                  public lsst::daf::data::LsstBase 
" {
"     ...
" };
"
" I would make the same plea to always allow classes to be written 
" with the opening brace on the following line, since I think it is 
" clearer.  But at least allowing it for the case of multiple 
" inheritances makes it possible for vim to indent it nicely
" without expanding the NicerCppIndent function far beyond my expertise.
"


" The basic tab size stuff:
setlocal shiftwidth=4
setlocal softtabstop=4
setlocal tabstop=4
setlocal expandtab

" This gets a lot of the indent stuff, but not namespace or template.
" Those are dealt with using indentexpr.  See below.
setlocal cinoptions=:2,l1,=5,g0,(0,u0,Ws,m1

" Turn on syntax:
syntax on
syntax match cTodo /\todo/

" Don't try to be vi
setlocal nocompatible

" Auto-wrap comments at 100 characters
setlocal formatoptions=crq
setlocal textwidth=100
setlocal comments=sr:/*,mb:*,ex:*/

" Highlight strings inside C comments
" This isn't required by any stretch, just something I prefer.
let c_comment_strings=1

" There are a few circumstances for which the cinoptions are not 
" sufficient to get the indenting correct.
" This function only checks for those expections and reverts to 
" cindent when that function is ok.
setlocal indentexpr=NicerCppIndent(v:lnum)

" First a helper function to find the line above the given line that
" contains code (i.e. not a comment).
" This procedure is not completely rigorous. 
" I've made some assumuptions about the comments not being ludicrous.
" e.g. A line like this isn't parsed correctly:
" /* */ some code /* */
" Also block comments that include a spurious /* will be parsed wrong.
" But any code like this deserves to be indented wrong...
function! FindPreviousCodeLine(lnum)
    let prevnum = a:lnum-1
    while prevnum > 0
        "Skip if line is all whitespace
        if getline(prevnum) =~ '^\s*$'
            let prevnum = prevnum-1
        "Skip if line is a comment
        elseif getline(prevnum) =~ '^\s*//'
            let prevnum = prevnum-1
        elseif getline(prevnum) =~ '^\s*/\*.*\*/\s*$'
            let prevnum = prevnum-1
        "Skip if line is a preprocessor directive
        elseif getline(prevnum) =~ '^\s*#'
            let prevnum = prevnum-1
        "If line is the end of a comment scan back to find the start
        elseif getline(prevnum) =~ '\*/\s*$'
            let prevnum = prevnum-1
            while !(getline(prevnum) =~ '/\*')
                let prevnum = prevnum-1
            endwhile
            "Decrement again if the comment starts the line
            if (getline(prevnum) =~ '^\s*/\*')
                let prevnum = prevnum-1
            endif
        "Otherwise we are at a code line, so return it
        else 
            return prevnum
        endif
    endwhile
    return 0
endfunction

" Now the actual function that returns the indent level for a given line:
function! NicerCppIndent(lnum)

    " This checks the name of the syntax at the current cursor position
    " If it contains the word Comment, then we are in a comment,
    " in which case, we can just use the normal cindent function
    " (Unless this is the start of the comment...)
    let firstS = match(getline(a:lnum),'\S')
    let synName = synIDattr(synID(a:lnum, 1, firstS), "name")
    let startComment = getline(a:lnum) =~ '^\s*/[/*]' 
    if (synName =~ 'Comment') && !startComment
        return cindent(a:lnum)
    endif
    
    let prevnum = FindPreviousCodeLine(a:lnum)

    "If no previous line, then cindent will work fine
    if prevnum == 0
        "(Unless this line is a template, in which case it might not.)
        if getline(a:lnum) =~ '^\s*\<template\>\s*<.*>\s*$'
            return cindent(a:lnum-1)
	else
            return cindent(a:lnum)
	endif
    endif
    
    "If the previous line ends with > and has no matching <, then it might be
    "a multi-line template specification.
    "This nlangle,nrangle business is to allow for template<> within
    "the template arguments.  See e.g. MaskedImageIteratorBase in afw/image.
    "I'm not sure if this covers the full range of possibilities, but it
    "seems to work on at least moderatley complex examples.
    let nlangle = count(split(getline(prevnum),'\zs'),'<')
    let nrangle = count(split(getline(prevnum),'\zs'),'>')
    if (getline(prevnum) =~ '>\s*$') && (nrangle > nlangle)
        let prevnum2 = FindPreviousCodeLine(prevnum)
	let nlangle2 = count(split(getline(prevnum2),'\zs'),'<')
	let nrangle2 = count(split(getline(prevnum2),'\zs'),'>')
	"Keep going up until we get to the start of the template.
	while (getline(prevnum2) =~ ',\s*$') && (nrangle2 > nlangle2)
            let prevnum2 = FindPreviousCodeLine(prevnum)
	endwhile
	"Make sure this is what we think it is.
	if getline(prevnum2) =~ '^\s*\<template\>\s*<.*,\s*$'
	    return indent(prevnum2)
	endif
    endif

    "If we are in the midst of a multi-line template specification
    "then align the later lines after the <
    if getline(prevnum) =~ '^\s*\<template\>\s*<.*,\s*$'
        return match(getline(prevnum),'<') + 1
    endif

    "If the previous line is template<...>, then keep same alignment
    "This should be done after the above check because the template could
    "be within part of a larger template specification.
    "e.g. MaskedImageIteratorBase again.
    if getline(prevnum) =~ '^\s*\<template\>\s*<.*>\s*$'
        return indent(prevnum)
    endif

    "cindent occasionally gets the indent wrong for the template line itself.
    "It gets it right at first as you are writing it, but after completing
    "the function declaration, it can sometimes want to indent it by an
    "extra shiftwidth.
    "I haven't found a correct way to get around this, but normally, the 
    "template line is preceded by a blank line, for which cindent 
    "returns the correct value.  
    "So if that's the case, return that value instead of cindent(a:lnum).
    if getline(a:lnum) =~ '^\s*\<template\>\s*<.*>\s*$'
        if getline(a:lnum-1) =~'^\w*$'
	    return cindent(a:lnum-1)
	endif
    endif

    "If the previous line is a class header with an unfinished list
    "of inheritances, then align at the : +2 spaces
    "Also if : starts the line, use same alignment rule.
    if getline(prevnum) =~ '^\s*\(\<class\>.*\)*:.*,\s*$'
    	return match(getline(prevnum),':') + 2
    endif

    "If this line is a nested namespace, only do a single indent:
    "(Make sure line doesn't end with ;, since then it is just a 
    " namespace alias, not the start of a new namespace.)
    if getline(a:lnum) =~ '^\s*\<namespace\>.*[^;]$'
        "If the previous line starts with namespace, then keep same alignment
        if getline(prevnum) =~ '^\s*\<namespace\>'
            return indent(prevnum)
        endif

        "In the LSST standard, the brace is supposed to be on the same line
        "as the namespace, but check for it anyway
        if getline(prevnum) =~ '^\s*{'
            let prevnum2 = FindPreviousCodeLine(prevnum)
            if getline(prevnum2) =~ '^\s*\<namespace\>'
                return indent(prevnum2)
            else
                return cindent(a:lnum)
            endif
        endif
    endif

    "Very occasionally cindent gets the indent wrong for an inheritance
    "line that starts after a class Blah : line.  (I think it only happens
    "for a template specialization, so class Blah<std::complex<T> > :
    "for example).  So check for this too.
    if getline(prevnum) =~ '^\s*\(\<class\>.*\)*:\s*$'
    	return indent(prevnum) + &sw
    endif

    "For other situations, just use the normal cindent function.
    return cindent(a:lnum)

endfunction

