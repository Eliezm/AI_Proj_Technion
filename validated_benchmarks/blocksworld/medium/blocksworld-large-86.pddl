(define (problem blocksworld-large-86)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b13 b14 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b6) (on-table b8) (on b1 b6) (on b10 b4) (on b11 b8) (on b12 b3) (on b13 b12) (on b14 b7) (on b2 b0) (on b3 b14) (on b4 b9) (on b5 b2) (on b7 b10) (on b9 b5) (clear b1) (clear b11) (clear b13) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on-table b12) (on-table b14) (on-table b3) (on-table b4) (on b1 b5) (on b10 b4) (on b11 b1) (on b13 b3) (on b2 b0) (on b5 b7) (on b6 b12) (on b7 b2) (on b8 b9) (on b9 b11) (clear b10) (clear b13) (clear b14) (clear b6) (clear b8) (arm-empty))
  )
)
