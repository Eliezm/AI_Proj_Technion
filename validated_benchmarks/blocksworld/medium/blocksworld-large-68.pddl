(define (problem blocksworld-large-68)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b13) (on-table b3) (on-table b6) (on b0 b1) (on b1 b2) (on b10 b0) (on b11 b6) (on b12 b14) (on b14 b5) (on b2 b12) (on b4 b8) (on b5 b3) (on b7 b10) (on b8 b13) (on b9 b4) (clear b11) (clear b7) (clear b9) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on-table b2) (on b10 b7) (on b11 b8) (on b12 b9) (on b13 b10) (on b14 b11) (on b3 b0) (on b4 b1) (on b5 b2) (on b6 b3) (on b7 b4) (on b8 b5) (on b9 b6) (clear b12) (clear b13) (clear b14) (arm-empty))
  )
)
