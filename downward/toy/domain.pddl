(define (domain sliding-puzzle)
  (:requirements :strips :typing)
  (:types tile pos)

  (:predicates
    (at       ?t - tile ?p - pos)    ; tile ?t is on position ?p
    (empty    ?p - pos)              ; position ?p is empty
    (adjacent ?p1 - pos ?p2 - pos)   ; ?p1 and ?p2 are neighboring positions
  )

  (:action move
    :parameters (?t    - tile
                 ?from - pos
                 ?to   - pos)
    :precondition (and
      (at       ?t    ?from)
      (empty    ?to)
      (adjacent ?from ?to)
    )
    :effect (and
      (not (at    ?t    ?from))
      (not (empty ?to))
      (at    ?t    ?to)
      (empty ?from)
    )
  )
)
