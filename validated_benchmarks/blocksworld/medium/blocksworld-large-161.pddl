(define (problem blocksworld-large-161)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b1) (on-table b12) (on-table b16) (on-table b6) (on-table b7) (on-table b9) (on b0 b2) (on b10 b0) (on b11 b15) (on b13 b5) (on b14 b16) (on b15 b14) (on b2 b8) (on b3 b7) (on b4 b1) (on b5 b9) (on b8 b6) (clear b10) (clear b11) (clear b12) (clear b13) (clear b3) (clear b4) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b11 b9) (on b12 b10) (on b13 b11) (on b14 b12) (on b15 b13) (on b16 b14) (on b2 b0) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b15) (clear b16) (arm-empty))
  )
)
