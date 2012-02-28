;;
;; Support code for the LSST C++ coding standards
;;
;; The basic tool is a "lsst-c++" mode derived from c++-mode:
;;   (load-file "/path/to/lsst.el")
;;   (lsst-c++-mode)
;;
;; If you want to always use this mode for C++, add this to your .emacs file:
;;   (load-file "/path/to/lsst.el")
;;   (lsst-c++-default)
;;
;; Even if you don't run lsst-c++-default, you might want to steal part
;; of its hook command (e.g. bind ESC-^A to go to the start of an
;; LSST-style function with the opening brace at the end of the line)

(define-derived-mode lsst-c++-mode
  c++-mode "LSST C++"
  "Major mode for editing LSST C++ source.
          \\{lsst-c++-mode-map}
"
  
  (let ( (our-mode-name mode-name) )
    (c++-mode)			        ; get correct font-lock behaviour
    (setq mode-name our-mode-name))	; keep our desired name

  (setq c-default-style '((other . "k&r")))
  (setq c-basic-offset 4)
  ;; Format switches correctly
  (let ( (c-offsets-alist '((member-init-intro . ++))) )
    (add-to-list 'c-offsets-alist (cons 'statement-case-intro 2))
    (add-to-list 'c-offsets-alist (cons 'statement-case-open 2))
    (add-to-list 'c-offsets-alist (cons 'case-label 2)))

  (setq comment-column 40)
  (setq fill-column 80)
  (setq tab-width 8)
  (setq indent-tabs-mode nil) ;; use spaces instead of tabs
  )
;;;
;;; The code standards state the files should specify C++-LSST mode;
;;; that's an error, but we'll let it work too.
;;;
(define-derived-mode c++-lsst-mode
  lsst-c++-mode "LSST C++" "Alias for lsst-c++-mode")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun lsst-c++-default (&optional delete)
  "Use LSST C++-mode as default for C++"

  (interactive)
  (add-to-list 'auto-mode-alist (cons "\\.\\(cc\\|h\\(pp\\)?\\)$"
				      'lsst-c++-mode))
  
  (add-hook 'lsst-c++-mode-hook
	    '(lambda ()
	       ;; Make ESC-^A/E go to the start/end of an LSST-style
	       ;; function with the opening brace at the end of the
	       ;; line
	       (define-key c-mode-base-map "\e\C-a" 'c-beginning-of-defun)
	       (define-key c-mode-base-map "\e\C-e" 'c-end-of-defun)
	    ))
  ;; This is a hack! It turns out that the first time that
  ;; we invoke lsst-c++-mode the closing brace of a global-scope
  ;; for/while loop is indented an extra space.
  ;;
  ;; I should figure our why, but here's a quick workaround.
  (if t
      (let ( (tmp "''''''hack.cpp") )
	(set-buffer (get-buffer-create tmp))
	(lsst-c++-mode)
	(kill-buffer tmp)))
  )
