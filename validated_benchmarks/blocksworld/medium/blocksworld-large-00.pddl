(define (problem blocksworld-large-0)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b14) (on-table b6) (on b1 b0) (on b10 b11) (on b11 b2) (on b12 b9) (on b13 b4) (on b2 b7) (on b3 b6) (on b4 b12) (on b5 b13) (on b7 b14) (on b8 b10) (on b9 b8) (clear b1) (clear b3) (clear b5) (arm-empty)
  )
  (:goal
    (and (on-table b10) (on-table b13) (on-table b14) (on-table b3) (on-table b5) (on b0 b13) (on b1 b11) (on b11 b7) (on b12 b8) (on b2 b1) (on b4 b5) (on b6 b12) (on b7 b14) (on b8 b2) (on b9 b10) (clear b0) (clear b3) (clear b4) (clear b6) (clear b9) (arm-empty))
  )
)
