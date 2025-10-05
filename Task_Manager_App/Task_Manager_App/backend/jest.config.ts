import type { Config } from 'jest';

const config: Config = {
  displayName: 'backend',
  preset: '../jest-preset.js',
  testEnvironment: 'node',
  transform: {
    '^.+\\.[tj]s$': 'ts-jest',
  },
  moduleFileExtensions: ['ts', 'js', 'json'],
  coverageDirectory: '../../coverage/backend',
  roots: ['<rootDir>/src'],
};

export default config;