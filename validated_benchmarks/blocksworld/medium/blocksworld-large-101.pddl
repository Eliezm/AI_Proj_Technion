(define (problem blocksworld-large-101)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b13) (on-table b6) (on-table b7) (on b1 b0) (on b10 b6) (on b11 b9) (on b12 b10) (on b14 b8) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b8 b5) (on b9 b7) (clear b11) (clear b12) (clear b13) (clear b14) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b13 b12) (on b14 b13) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b14) (arm-empty))
  )
)
