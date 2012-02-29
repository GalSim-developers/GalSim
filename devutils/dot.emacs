;; LSST
(load "lsst")
(load "lsst-utils")
(let ( (lsst t) (width 100) )
  (if lsst
      (lsst-c++-default))

  (if (and lsst window-system)
      (progn
        (set-default 'fill-column width)
        (set-frame-width default-minibuffer-frame (1+ width))
        ;; ` quotes, but evaluates ,width
        (add-hook 'after-make-frame-functions
                  `(lambda (frame)
                     (set-frame-width frame (1+ ,width))))
        ))
  )

(add-hook 'lsst-c++-mode-hook
          '(lambda ()
             (define-key c-mode-base-map "\e\C-a" 'c-beginning-of-defun)
             (define-key c-mode-base-map "\e\C-e" 'c-end-of-defun)))
(add-hook 'lsst-c++-mode-hook
          '(lambda ()
             (define-key c-mode-base-map "\C-c>" 'indent-region)
             (define-key c-mode-base-map "\C-c<" 'indent-region)))

;(add-hook 'lsst-c++-mode-hook 'lsst-set-width)
