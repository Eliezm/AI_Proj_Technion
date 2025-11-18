(define (problem blocksworld-large-4)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b2 b20 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b13) (on-table b14) (on-table b15) (on-table b19) (on-table b3) (on-table b7) (on b1 b10) (on b10 b16) (on b11 b3) (on b12 b18) (on b16 b6) (on b17 b19) (on b18 b13) (on b2 b0) (on b20 b15) (on b4 b2) (on b5 b7) (on b6 b4) (on b8 b9) (on b9 b14) (clear b1) (clear b11) (clear b12) (clear b17) (clear b20) (clear b5) (clear b8) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b1) (on b10 b8) (on b11 b9) (on b12 b10) (on b13 b11) (on b14 b12) (on b15 b13) (on b16 b14) (on b17 b15) (on b18 b16) (on b19 b17) (on b2 b0) (on b20 b18) (on b3 b1) (on b4 b2) (on b5 b3) (on b6 b4) (on b7 b5) (on b8 b6) (on b9 b7) (clear b19) (clear b20) (arm-empty))
  )
)
