;; ==============================
;; puzzle8_domain.pddl
;; ==============================
(define (domain puzzle)
  (:requirements :strips :typing)
  (:types tile position)

  (:predicates
    (at ?tile - tile ?pos - position)
    (empty ?pos - position)
  )

  (:action move
    :parameters (?tile - tile ?from - position ?to - position)
    :precondition (and
      (at ?tile ?from)
      (empty ?to)
    )
    :effect (and
      ;; tile leaves its old spot
      (not (at ?tile ?from))
      ;; tile arrives in the new spot
      (at ?tile ?to)
      ;; new spot is no longer empty, old spot becomes empty
      (not (empty ?to))
      (empty ?from)
    )
  )
)