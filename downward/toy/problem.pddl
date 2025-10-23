(define (problem eight-puzzle)
  (:domain sliding-puzzle)

  (:objects
    tile1 tile2 tile3 tile4 tile5 tile6 tile7 tile8 - tile
    p1 p2 p3 p4 p5 p6 p7 p8 p9             - pos
  )

  (:init
    ;; initial configuration (you can scramble these as you like)
    (at tile1 p1) (at tile2 p2) (at tile3 p3)
    (at tile4 p4) (at tile5 p5) (at tile6 p6)
    (at tile7 p7) (at tile8 p8)
    (empty p9)

    ;; horizontal adjacency
    (adjacent p1 p2) (adjacent p2 p3)
    (adjacent p4 p5) (adjacent p5 p6)
    (adjacent p7 p8) (adjacent p8 p9)

    ;; vertical adjacency
    (adjacent p1 p4) (adjacent p2 p5) (adjacent p3 p6)
    (adjacent p4 p7) (adjacent p5 p8) (adjacent p6 p9)

    ;; make adjacency bidirectional
    (adjacent p2 p1) (adjacent p3 p2)
    (adjacent p5 p4) (adjacent p6 p5)
    (adjacent p8 p7) (adjacent p9 p8)
    (adjacent p4 p1) (adjacent p5 p2) (adjacent p6 p3)
    (adjacent p7 p4) (adjacent p8 p5) (adjacent p9 p6)
  )

  (:goal (and
    (at tile1 p1) (at tile2 p2) (at tile3 p3)
    (at tile4 p4) (at tile5 p5) (at tile6 p6)
    (at tile7 p7) (at tile8 p8) (empty p9)
  ))
)
