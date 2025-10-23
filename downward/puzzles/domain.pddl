(define (domain sliding-tile)
  (:predicates
    (at ?t ?c)        ; tile ?t is at cell ?c
    (empty ?c)        ; the blank is at cell ?c
    (adj ?c1 ?c2)     ; cells are adjacent (orthogonal)
  )
  (:action slide
    :parameters (?t ?from ?to)
    :precondition (and (at ?t ?from) (empty ?to) (adj ?from ?to))
    :effect (and
      (at ?t ?to)
      (empty ?from)
      (not (at ?t ?from))
      (not (empty ?to))
    )
  )
)
