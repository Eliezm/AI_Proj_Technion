(define (problem blocksworld-large-23)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b11) (on-table b12) (on-table b5) (on-table b6) (on b0 b6) (on b1 b5) (on b10 b1) (on b13 b11) (on b14 b8) (on b2 b4) (on b3 b7) (on b4 b9) (on b7 b14) (on b8 b12) (on b9 b3) (clear b0) (clear b10) (clear b13) (clear b2) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b13 b12) (on b14 b13) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b14) (arm-empty))
  )
)
