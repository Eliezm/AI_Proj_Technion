(define (problem blocksworld-large-2)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b11) (on-table b3) (on-table b6) (on-table b9) (on b1 b3) (on b10 b13) (on b12 b7) (on b13 b14) (on b14 b9) (on b2 b6) (on b4 b2) (on b5 b10) (on b7 b4) (on b8 b0) (clear b1) (clear b11) (clear b12) (clear b5) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b11 b9) (on b12 b10) (on b13 b11) (on b14 b12) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b13) (clear b14) (arm-empty))
  )
)
