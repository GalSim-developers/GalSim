;;
;; Utilities to help with coding for LSST
;;
(defun lsst-strip-namespaces ()
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward "lsst::afw::\\|boost::\\|gil::" nil t)
      (replace-match ""))))

(defun lsst-strip-allocator ()
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward ", *std::allocator< *\\([^>]+\\|<[^>]+>\\)  *>" nil t)
      (replace-match ""))))

(defun lsst-strip-template-arguments ()
  "Delete the template arguments around dot"
  (interactive)

  (let ( (start (point))
	 (init t)
	 (depth 0) )

    (skip-chars-backward "^<>")
    (if (= (point-min) (point))
	(error "You don't appear to be within a template argument"))

    (forward-char -1)
    (if (looking-at "<")
	(progn
	  (forward-char 1)
	  (setq start (point)))
      (goto-char start)
      (error "You don't appear to be within a template argument"))
  
    (while (or init (>= depth 0))
      (setq init nil)
      (if (not (re-search-forward "[<>]" nil t))
	  (error "I can't find the closing >"))
      (save-excursion
	(forward-char -1)
	(if (looking-at "<")
	    (setq depth (+ depth 1)))
	(if (looking-at ">")
	    (setq depth (- depth 1))))
      )
    (delete-region start (- (point) 1))))

(defun lsst-exception (extend)
  "Throw an LSST exception;  with prefix argument rethrow an exception"

  (interactive "P")

  (let ( (start (point)) what throw_macro_name)
    (if extend
	(progn
	  (setq throw_macro_name "LSST_EXCEPT_ADD")
	  (setq what (read-string "Name of exception to be rethrown: " "e"))
	  )
      (setq throw_macro_name "LSST_EXCEPT")
      (setq what (completing-read "Exception Type: " (list
						      "LogicErrorException"
						      "DomainErrorException"
						      "InvalidParameterException"
						      "LengthErrorException"
						      "OutOfRangeException"
						      "RuntimeErrorException"
						      "RangeErrorException"
						      "OverflowErrorException"
						      "UnderflowErrorException"
						      "NotFoundException"
						      "MemoryException"
						      "IoErrorException"
						      "TimeoutException"
						      "other")))
      )
    (if (string= what "other")
	(progn
	  (setq what (read-string "Type of exception: "))
	  (if (not what)
	      (error "You must specify a type"))
	  )
      (let ( (prefix "lsst::pex::exceptions::") )
	(if (not (string-match what (concat "^" prefix)))
	    (setq what (concat prefix what)))))

    (let ( (start (point)) )
      (insert (format "throw %s(%s,
                                (boost::format(\"Hello %%s\")
                                               %% \"world\"
                                               ).str());
" throw_macro_name what))
      (indent-region start (point))
      (goto-char start)	 
      (forward-line 1) (end-of-line)
      )
  ))
